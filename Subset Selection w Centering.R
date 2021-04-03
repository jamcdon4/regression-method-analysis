require(MASS)
require(faraway)
require(ISLR)
require(caret)
require(leaps)
require(glmnet)
require(boot)


concrete=data.frame(read.csv("concrete.csv"))
View(concrete)

# calculate centered predictors
cement_c=concrete$cement-mean(concrete$cement)
slag_c=concrete$slag-mean(concrete$slag)
fly_c=concrete$fly_ash-mean(concrete$fly_ash)
wat_c=concrete$water-mean(concrete$water)
sup_c=concrete$superplasticizer-mean(concrete$superplasticizer)
cour_c=concrete$course_aggregate-mean(concrete$course_aggregate)
fine_c=concrete$fine_aggregate-mean(concrete$fine_aggregate)
age_c=concrete$age-mean(concrete$age)

# add squares and second-order interactions to data frame
concrete$cem_sq=cement_c*cement_c
concrete$slg_sq=slag_c*slag_c
concrete$fly_sq=fly_c*fly_c
concrete$wat_sq=wat_c*wat_c
concrete$sup_sq=sup_c*sup_c
concrete$cour_sq=cour_c*cour_c
concrete$fine_sq=fine_c*fine_c
concrete$age_sq=age_c*age_c


concrete$cm_x_sl=cement_c*slag_c
concrete$cm_x_fl=cement_c*fly_c
concrete$cm_x_wt=cement_c*wat_c
concrete$cm_x_sp=cement_c*sup_c
concrete$cm_x_cor=cement_c*cour_c
concrete$cm_x_fn=cement_c*fine_c
concrete$cm_x_ag=cement_c*age_c

concrete$sl_x_fl=slag_c*fly_c
concrete$sl_x_wt=slag_c*wat_c
concrete$sl_x_sp=slag_c*sup_c
concrete$sl_x_cor=slag_c*cour_c
concrete$sl_x_fn=slag_c*fine_c
concrete$sl_x_ag=slag_c*age_c

concrete$fl_x_wt=fly_c*wat_c
concrete$fl_x_sp=fly_c*sup_c
concrete$fl_x_cor=fly_c*cour_c
concrete$fl_x_fn=fly_c*fine_c
concrete$fl_x_ag=fly_c*age_c

concrete$wt_x_sp=wat_c*sup_c
concrete$wt_x_cor=wat_c*cour_c
concrete$wt_x_fn=wat_c*fine_c
concrete$wt_x_ag=wat_c*age_c

concrete$sp_x_cor=sup_c*cour_c
concrete$sp_x_fn=sup_c*fine_c
concrete$sp_x_ag=sup_c*age_c

concrete$cor_x_fn=cour_c*fine_c
concrete$cor_x_ag=cour_c*age_c

concrete$fn_x_ag=fine_c*age_c



#a) Spliting the dataset into train and test set
data1 = concrete
set.seed(123)
train=sample(1:1030,806,replace=F)
train1=data1[train,]
test1=data1[-train,]

#b)
best.mods=regsubsets(concrete$strength~.,data=concrete,nvmax=44,method="exhaustive")
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
cv.err=matrix(NA,k,44,dimnames=list(NULL,paste(1:44)))

# perform CV
for (j in 1:k){
  # pick models with lowest RSS with 1-17 predictors fit without kth fold
  best.mods=regsubsets(strength~.,data=train1[folds!=j,],
                       nvmax=44,method="exhaustive")
  # estimate test error for all 17 models by predicting kth fold 
  for (i in 1:44){
    pred=pred.sbs(best.mods,train1[folds==j,],id=i)
    cv.err[j,i]=mean((train1$strength[folds==j]-pred)^2)  # save error est
  }
}

mse.cv=apply(cv.err,2,mean) # compute mean MSE for each number of predictors
min=which.min(mse.cv)  # find minimum mean MSE
mse.cv[min]

# plot and put a red circle around lowest MSE
par(mfrow=c(1,1))
plot(1:44,mse.cv,type="l",xlab="no. of predictors)",ylab="est. test MSE")
points(min,mse.cv[min],cex=2,col="red",lwd=2)
abline(h=c(0,50,100,150,200),lty=3)

#subset summary and selection
regfit.full1=regsubsets (strength~.,data=train1 ,nvmax =44)
regfit.summary=summary(regfit.full1)
regfit.summary
coef(regfit.full1,44)


#Selecting predictors for test set
best.mods1=regsubsets(train1$strength~.,data=train1,nvmax=44,method="exhaustive")
summary(best.mods1)
coef(best.mods1,44)

fit1=lm(test1$strength~.,data=test1)
par(mfrow=c(2,2))
plot(fit1)  # residual diagnostics
summary(fit1) # fit summary
vif(fit1)

#Finding MSE for test data set
predictcv=predict(fit1, newx=test1)
msecv=mean((predictcv-test1[,"strength"])^2)
msecv

