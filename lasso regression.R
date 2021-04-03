require(ISLR)
require(glmnet)
require(faraway)
require(dplyr)
require(leaps)

#setwd()

#read data
concrete=data.frame(read.csv("concrete.csv"))
View(concrete)

#centered predictors
cement_c=concrete$cement-mean(concrete$cement)
slag_c=concrete$slag-mean(concrete$slag)
fly_c=concrete$fly_ash-mean(concrete$fly_ash)
water_c=concrete$water-mean(concrete$water)
super_c=concrete$superplasticizer-mean(concrete$superplasticizer)
course_c=concrete$course_aggregate-mean(concrete$course_aggregate)
fine_c=concrete$fine_aggregate-mean(concrete$fine_aggregate)
age_c=concrete$age-mean(concrete$age)

# add square to data frame
concrete$cement_sq=cement_c*cement_c
concrete$slag_sq=slag_c*slag_c
concrete$fly_sq=fly_c*fly_c
concrete$water_sq=water_c*water_c
concrete$super_sq=super_c*super_c
concrete$course_sq=course_c*course_c
concrete$fine_sq=fine_c*fine_c
concrete$age_sq=age_c*age_c

#add second-order interactions to data frame
concrete$c_s_sq=cement_c*slag_c
concrete$c_f_sq=cement_c*fly_c
concrete$c_w_sq=cement_c*water_c
concrete$c_su_sq=cement_c*super_c
concrete$c_c_sq=cement_c*course_c
concrete$c_fi_sq=cement_c*fine_c
concrete$c_a_sq=cement_c*age_c

concrete$s_f_sq=slag_c*fly_c
concrete$s_w_sq=slag_c*water_c
concrete$s_su_sq=slag_c*super_c
concrete$s_c_sq=slag_c*course_c
concrete$s_fi_sq=slag_c*fine_c
concrete$s_a_sq=slag_c*age_c

concrete$f_w_sq=fly_c*water_c
concrete$f_su_sq=fly_c*super_c
concrete$f_c_sq=fly_c*course_c
concrete$f_fi_sq=fly_c*fine_c
concrete$f_a_sq=fly_c*age_c

concrete$w_su_sq=water_c*super_c
concrete$w_c_sq=water_c*course_c
concrete$w_fi_sq=water_c*fine_c
concrete$w_a_sq=water_c*age_c

concrete$su_c_sq=super_c*course_c
concrete$su_f_sq=super_c*fine_c
concrete$su_a_sq=super_c*age_c

concrete$cr_f_sq=course_c*fine_c
concrete$cr_a_sq=course_c*age_c

concrete$fi_a_sq=fine_c*age_c


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

