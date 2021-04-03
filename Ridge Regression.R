require(MASS)
require(ISLR)
require(caret)
require(glmnet)
require(faraway)
require(leaps)
require(boot)

#Importing Data
concrete=read.csv("concrete.csv")

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
