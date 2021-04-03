require(MASS)
require(faraway)
require(ISLR)
require(caret)
require(leaps)
require(glmnet)
require(boot)


concrete=data.frame(read.csv("concrete.csv"))
View(concrete)

#a) Spliting the dataset into train and test set
data1 = concrete
set.seed(123)
train=sample(1:1030,806,replace=F)
train1=data1[train,]
test1=data1[-train,]

#b)
best.mods=regsubsets(concrete$strength~.,data=concrete,nvmax=8,method="exhaustive")
summary(best.mods)

# use 5-fold CV to determine which best model for each number of predictors
# has the lowest estimated test error

# create function to predict from reg subsets object

pred.sbs=function(obj,new,id){
  form=as.formula(obj$call[[2]])
  mat=model.matrix(form,new)
  coefi=coef(obj,id=id)
  xvars=names(coefi)
  return(mat[,xvars]%*%coefi)
}

# set up for cross validation

k=5  # set number of folds
set.seed(123)
# create an index with id 1-5 to assign observations to folds
folds=sample(1:k,nrow(train1),replace=T) 
# create dummy matrix to store CV error estimates
cv.err=matrix(NA,k,8,dimnames=list(NULL,paste(1:8)))

# perform CV
for (j in 1:k){
  # pick models with lowest RSS with 1-17 predictors fit without kth fold
  best.mods=regsubsets(strength~.,data=train1[folds!=j,],
                       nvmax=8,method="exhaustive")
  # estimate test error for all 17 models by predicting kth fold 
  for (i in 1:8){
    pred=pred.sbs(best.mods,train1[folds==j,],id=i)
    cv.err[j,i]=mean((train1$strength[folds==j]-pred)^2)  # save error est
  }
}

mse.cv=apply(cv.err,2,mean) # compute mean MSE for each number of predictors
min=which.min(mse.cv)  # find minimum mean MSE
mse.cv[min]

# plot and put a red circle around lowest MSE
par(mfrow=c(1,1))
plot(1:8,mse.cv,type="b",xlab="no. of predictors)",ylab="est. test MSE",ylim=c(0,400))
points(min,mse.cv[min],cex=2,col="red",lwd=2)
abline(h=c(100,200,300),lty=3)

#subset summary and selection
regfit.full1=regsubsets (strength~.,data=train1 ,nvmax =19)
regfit.summary=summary(regfit.full1)
regfit.summary
coef(regfit.full1,8)


#Selecting predictors for test set
best.mods1=regsubsets(train1$strength~.,data=train1,nvmax=8,method="exhaustive")
summary(best.mods1)
coef(best.mods1,8)

#We get 11 significant predictors 
# They are Accept,Top10perc,Expend,Private,Room.Board,PhD,Top25perc,perc.alumni,Grad.Rate,Terminal,F.Undergrad

#Running model on test data set
fit1=lm(test1$strength~.,data=test1)
par(mfrow=c(2,2))
plot(fit1)  # residual diagnostics
summary(fit1) # fit summary
vif(fit1)

#Finding MSE for test data set
predictcv=predict(fit1, newx=test1)
msecv=mean((predictcv-test1[,"strength"])^2)
msecv
