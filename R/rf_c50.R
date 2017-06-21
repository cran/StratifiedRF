#' @import C50
#' @import dplyr
#' @importFrom stats as.formula
`%>%` <- magrittr::`%>%`


#' Stratified Random Forest
#'
#' Random Forest that works with groups of predictor variables. When building a tree, a number of variables is taken from each group separately. Useful when rows contain information about different things (e.g. user information and product information) and it's not sensible to make a prediction with information from only one group of variables, or when there are far more variables from one group than the other and it's desired to have groups appear evenly on trees.
#'
#' Trees are grown using the C5.0 algorithm. Implementation of everything outside the tree-building is in native R code, thus might be slow. Currently works for classification only.

#' @param df Data to build the model (data.frame only).
#' @param targetvar String indicating the name of the target or outcome variable in the data.
#' @param groups Unnamed list containing, at each entry, a group of variables (as a string vector with their names).
#' @param mtry A numeric vector indicating how many variables to take from each group when building each tree.
#' @param ntrees Number of trees to grow.
#' @param class_quotas How many rows from each class to use in each tree (useful when there is a class imbalance). Must be a numeric vector or a named list with the number of desired rows to sample for each level of the target variable. Note that using more rows than the data originally had might result in incorrect out-of-bag error estimates.
#' @param fulldepth Whether to grow the trees to full depth.
#' @param replacement Whether to sample rows with replacement.
#' @keywords stratified_rf
#' @export
#' @examples
#' data(iris)
#' groups <- list(c("Sepal.Length","Sepal.Width"),c("Petal.Length","Petal.Width"))
#' mtry <- c(1,1)
#' m <- stratified_rf(iris,"Species",groups,mtry,ntrees=2)
#' summary(m)
stratified_rf=function(df,targetvar,groups,mtry,ntrees=500,class_quotas=NULL,fulldepth=TRUE,replacement=TRUE){
  if (class(df)!="data.frame"){stop("Data to predict must be a data.frame")}
  if (class(targetvar)!="character"){stop('Target variable must be specified as a string')}
  if (class(mtry)!="numeric"){stop('mtry must be a vector or list of numbers')}
  if (class(groups)!="list"){stop('Predictor variables must be specified as an unnamed list of character vectors')}
  if (class(groups[[1]])!="character"){stop('Predictor variables must be specified as an unnamed list of character vectors')}
  if ((class(ntrees)!="numeric")|(length(ntrees)!=1)){stop('ntrees must be a positive integer')}
  if (!is.null(class_quotas)){
    if (class(class_quotas)!='numeric' & class(class_quotas)!='list'){stop('class_quotas must be a numeric vector or list with one entry per level of the target variable')}
    levs_target=levels(eval(parse(text=paste0("df$",targetvar))))
    len_target=length(levs_target)
    if (length(class_quotas)!=len_target){stop('class_quotas must be a numeric vector or list with one entry per level of the target variable')}}
    if (class(class_quotas)=='numeric'){
        class_quotas=as.list(class_quotas)
        names(class_quotas)=levs_target
    }

  tree.models=vector(mode="list", length=ntrees)
  tree.models.acc=vector(mode="list", length=ntrees)
  num.groups=length(groups)
  rowset=1:dim(df)[1]

  levels.target=levels(eval(parse(text=paste0("df$",targetvar))))
  oob.sumprob=data.frame(row.names=rowset)
  for (l in levels.target){
    oob.sumprob[l]=rep(0,dim(df)[1])
  }
  oob.times=rep(0,dim(df)[1])
  treecontrol=C5.0Control(noGlobalPruning = fulldepth)

  if (!is.null(class_quotas)){
    rowset.classes=list()
    for (i in levels.target){
      rowset.classes[[i]]=which(df[,targetvar]==i)
    }
  }

  for (i in 1:ntrees){
    vars=vector()
    for (j in 1:length(groups)){
      vars=c(vars,sample(groups[[j]],size=mtry[j],replace=FALSE))
    }

    if (is.null(class_quotas)){
    trainrows=sample(rowset,replace=replacement)
    } else {
    trainrows=list()
    for (l in levels.target){trainrows[[l]]=sample(rowset.classes[[l]],size=class_quotas[[l]],replace=TRUE)}
    trainrows=unlist(trainrows)
    }

    oob.rows=!(rowset %in% trainrows)

    traindata=df[trainrows,c(vars,targetvar)]
    form=as.formula(paste(targetvar, paste(vars, collapse=" + "), sep=" ~ "))

    tree=C5.0(form,data=traindata,control=treecontrol,rules=FALSE,trials=1)
    tree.models[[i]]=tree

    oob.sumprob[oob.rows,]=oob.sumprob[oob.rows,]+predict.C5.0(tree.models[[i]],df[oob.rows,],type='prob')
    oob.times[oob.rows]=oob.times[oob.rows]+1

    ########## this is new ############
    tree.models.acc[[i]]=mean(df[oob.rows,targetvar]==predict.C5.0(tree.models[[i]],df[oob.rows,],type='class'))

  }

  probs=oob.sumprob/oob.times
  pred=levels.target[max.col(probs)]
  real=eval(parse(text=paste0("df$",targetvar)))

  acc=mean(pred==real,na.rm = TRUE)
  conf_matrix=table(real,pred)

  return(structure(list(trees=tree.models,acc.trees=unlist(tree.models.acc),oob.preds=probs,acc=acc,conf_matrix=conf_matrix,vargroups=groups,levels_target=levels.target,targetvar=targetvar),class='stratified_rf'))
}


#' Make predictions on new data
#'
#' Make predictions from a stratified_rf model on new data.
#'
#' Note that by default the predictions are made quite differently from the original Random Forest algorithm.

#' @param object A stratified_rf model.
#' @param data New data on which to make predictions (data.frame only). Must have the same names as the data used to build the model.
#' @param type Prediction type. Either "class" to get the predicted class or "prob" to get the voting scores for each class.
#' @param agg_type How to take the final prediction from those of each separate tree. Either "prob" to average the probabilities output from each tree or "class" to count the final predictions from each.
#' @param vote_type How to weight the outputs from each tree. Either "simple" to average them, or "weighted" for a weighted average according to their OOB classification accuracy.
#' @param ... other options (not currently used)
#' @keywords predict.stratified_rf
#' @export
#' @examples
#' data(iris)
#' groups <- list(c("Sepal.Length","Sepal.Width"),c("Petal.Length","Petal.Width"))
#' mtry <- c(1,1)
#' m <- stratified_rf(iris,"Species",groups,mtry,ntrees=2)
#' predict(m,iris)
predict.stratified_rf=function(object,data,type='class',agg_type='prob',vote_type='simple',...){
  model=object
  if (!(type %in% c('class','prob','raw'))){stop("prediction type must be 'prob' or 'class'")}
  if (!(agg_type %in% c('class','prob'))){stop("aggregation type must be 'prob' or 'class'")}
  if (!(vote_type %in% c('simple','weighted'))){stop("vote type must be either simple or weighted")}
  if (class(data)!="data.frame"){stop("Data to predict must be a data.frame")}
  if (sum(!(unlist(model$vargroups) %in% names(data)))>0){stop("Data to predict doesn't have the same variables")}

  if (agg_type=='prob') {
    if (vote_type=='simple'){preds=Reduce('+',lapply(model$trees,predict.C5.0,data,type='prob'))}
    if (vote_type=='weighted'){
      preds=lapply(model$trees,predict.C5.0,data,type='prob')
      preds2=lapply(seq_along(preds),function(x) model$acc.trees[x]*preds[[x]])
      preds=Reduce('+',preds2)
    }
    if (type=='class'){return(model$levels_target[max.col(preds)])}
    if ((type=='prob')|(type=='raw')){return(preds/apply(preds,1,sum))}
  }

  if (agg_type=='class') {
    votes=Reduce(cbind,lapply(model$trees,predict.C5.0,data,type='class'))

    if (vote_type=='simple'){
      if (type=='class'){return(model$levels_target[as.numeric(apply(votes,1,function(x) names(sort(-table(x)))[1]))])}
      if ((type=='prob')|(type=='raw')){
        vote_counts=data.frame(row.names=1:dim(data)[1])
        for (i in model$levels_target){vote_counts[,i]=apply(votes,1,function(x) sum(model$levels_target[x]==i))}
        return(vote_counts/apply(vote_counts,1,sum))
      }}

    if (vote_type=='weighted'){
      vote_counts=data.frame(row.names=1:dim(data)[1])
      for (i in model$levels_target){
        agg_func=function(x){
          cnt=(model$levels_target[x]==i)*model$acc.trees[seq_along(x)]
          return(sum(cnt))
        }
        vote_counts[,i]=apply(votes,1,agg_func)}

      if (type=='class'){return(model$levels_target[max.col(vote_counts)])}
      if ((type=='prob')|(type=='raw')){vote_counts/apply(vote_counts,1,sum)}
    }
  }
}

#' Heuristic on variable importance
#'
#' Heuristic on variable importance, taken as averages from the variable importances calculated for each tree.
#'
#' Methods are taken directly from the C5.0 trees.

#' @param model A stratified_rf model.
#' @param metric How to calculate the variable importance from each tree. Either "usage" or "splits".
#' @export
#' @examples
#' data(iris)
#' groups <- list(c("Sepal.Length","Sepal.Width"),c("Petal.Length","Petal.Width"))
#' mtry <- c(1,1)
#' m <- stratified_rf(iris,"Species",groups,mtry,ntrees=2)
#' varimp_stratified_rf(m)
#' @return A named data frame with the importance score of each variable, sorted from largest to smallest.
varimp_stratified_rf=function(model,metric='usage'){
  if (class(model)!='stratified_rf'){stop("Model is not a stratified_rf")}
  ntrees=length(model$trees)
  reduce.func=function(df1,df2){
    df1$key=row.names(df1)
    df2$key=row.names(df2)

    sum1=sum2=Overall=NULL
    res=df1 %>% rename(sum1=Overall) %>% full_join(df2 %>% rename(sum2=Overall),by=c('key'='key')) %>% rowwise() %>% mutate(Overall=sum(sum1,sum2,na.rm=TRUE))
    return(data.frame(row.names=res$key,Overall=res$Overall))
  }
  lst.varimps=lapply(model$trees,C5imp,metric=metric,pct=FALSE)
  imps=Reduce(reduce.func,lst.varimps)/ntrees
  return(imps[order(-imps$Overall),,drop=FALSE])
}

#' Print summary statistics from a model
#' @param x A stratified_rf model.
#' @param ... other options (not currently used)
#' @export
#' @examples
#' data(iris)
#' groups <- list(c("Sepal.Length","Sepal.Width"),c("Petal.Length","Petal.Width"))
#' mtry <- c(1,1)
#' m <- stratified_rf(iris,"Species",groups,mtry,ntrees=2)
#' print(m)
print.stratified_rf=function(x,...){
  model=x
  asperc=function(x){paste0(substr(round(100*x,2),1,4),'%')}
  cat('Stratified Random Forest Model\n\n')
  cat('Out-of-bag prediction error: ',asperc(1-model$acc))
  cat('\n\nConfusion Matrix\n')
  tbl_errs=model$conf_matrix
  print(tbl_errs)
  cat('\n')
  class.errs=vector()
  for (i in 1:length(model$levels_target)){class.errs=c(class.errs,1-tbl_errs[i,i]/sum(tbl_errs[i,]))}
  for (i in 1:length(model$levels_target)){
    prec=tbl_errs[i,i]/sum(tbl_errs[,i])
    rec=tbl_errs[i,i]/sum(tbl_errs[i,])
    cat('Class ',model$levels_target[i],'- Precision:',asperc(prec),'Recall:',asperc(rec),'\n')
  }
  cat('\nPredictor Variables:\n')
  for (i in 1:length(model$vargroups)){
    cat('Group',i,': ',paste(model$vargroups[[i]],collapse=', '),'\n')
  }
  cat('\nTarget Variable: ',model$targetvar)
}

#' Summary statistics from a model
#' @param object A stratified_rf model.
#' @param ... other options (not currently used)
#' @export
#' @examples
#' data(iris)
#' groups <- list(c("Sepal.Length","Sepal.Width"),c("Petal.Length","Petal.Width"))
#' mtry <- c(1,1)
#' m <- stratified_rf(iris,"Species",groups,mtry,ntrees=2)
#' summary(m)
summary.stratified_rf=function(object,...){
  print.stratified_rf(object)
}
