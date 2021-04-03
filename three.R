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

#####################################################################################
#                                                                                   #
#                            Subset Selection                                       #
#                                                                                   #
#####################################################################################

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

#####################################################################################
#                                                                                   #
#                               Ridge Regression                                    #
#                                                                                   #
#####################################################################################

#Spliting the dataset into train and test set
data1 = concrete
set.seed(123)
train=sample(1:1030,806,replace=F)
train1=data1[train,]
test1=data1[-train,]

#Ridge Regression
# Creating Training and test Matrices
train.mat = model.matrix(strength~., data=train1)
test.mat = model.matrix(strength~., data=test1)

# Creating grid for lambda and fitting model using all lambdas
grid = 10 ^ seq(10, -2, length=100)
ridge.mod = glmnet(train.mat, train1[, "strength"], alpha = 0, lambda = grid)

# Coefficient values are plotted as lambda values are changes
plot(ridge.mod,xlab="L2 Norm")  
abline(h=0,lty=3)

#Model us executed on the training data with lambda grid
mod.ridge = cv.glmnet(train.mat, train1[, "strength"], alpha=0, lambda=grid, thresh=1e-12)
plot(mod.ridge)
mse.r=min(mod.ridge$cvm)
lambda.best = mod.ridge$lambda.min
lambda.best
mse.r

#Refitting the model with the best lambda
bm=glmnet(test.mat,test1[,"strength"],alpha = 0)


#Best model and OLS coefficients are compared
bestmodel=predict(bm,s=lambda.best,type="coefficients")
bestmodel

#Fitting the linear model
fit2=lm(strength~.,data=concrete)
slm=summary(lm(strength~.,data=concrete))
coef(slm)

#Plotting fitted values for Ridge and OLS
fit.ridge=predict(ridge.mod,s=lambda.best,test.mat)
plot(fit2$fitted.values,concrete$strength,pch=19,col="blue")
points(fit.ridge,test1$strength,col="red",lwd=2)
abline(a=0,b=1)

#MSE for Ridge Regression
msemdl=predict(bm, s=lambda.best, newx=test.mat)
mserr=mean((test1[,"strength"]-msemdl)^2)
mserr


#####################################################################################
#                                                                                   #
#                               Lasso Regression                                    #
#                                                                                   #
#####################################################################################


# remove observations with missing response values
concrete=concrete %>% filter(is.na(strength)==F)

# create index for random sample of 200
set.seed(123)
indx=sample(1:dim(concrete)[1],200,replace=F)

# create predictor matrix and vector for response

x=model.matrix(strength~.,concrete[indx,])[,-1] #- intercept column
y=concrete$strength[indx]

# create grid for lambda, fit model using all lambdas
grid=10^seq(2,-3,length=100) # lambda ranges from 100 to 0.001 
lasso.mod=glmnet(x,y,alpha=1,lambda=grid)  

# check coefficent values for each value of lambda
plot(lasso.mod)  # x-axis is in terms of sum(beta^2)
abline(h=0,lty=3)

# optimize lambda using cross-validation
set.seed(123)
cv.lasso=cv.glmnet(x,y,alpha=1,lambda=grid)
plot(cv.lasso)
bestlam.l=cv.lasso$lambda.min
mse.l=min(cv.lasso$cvm)
bestlam.l
mse.l

# best mse = 80 at lambda = 0.33

# get coefficents for best model and compare to OLS
lasso.coef=predict(lasso.mod,type="coefficients",s=bestlam.l)
lasso.coef

# plotfitted values for Ridge, compare with actual
fit.lasso=predict(lasso.mod,s=bestlam.l,x)
plot(fit.lasso,y,col="blue",lwd=2,pch=19)
abline(a=0,b=1)

R2.lasso=cor(fit.lasso,y)^2
R2.lasso
