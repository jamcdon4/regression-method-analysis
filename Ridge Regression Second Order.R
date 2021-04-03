require(MASS)
require(faraway)
require(ISLR)
require(caret)
require(leaps)
require(glmnet)
require(boot)

#Importing Data
concrete=read.csv("concrete.csv")

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