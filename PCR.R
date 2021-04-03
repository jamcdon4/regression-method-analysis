require(ISLR)
require(glmnet)
require(faraway)
require(dplyr)
require(leaps)
require(pls)

# read data and explore
concrete=data.frame(read.csv("concrete.csv"))
View(concrete)

# calculate centered predictors
cement_c=concrete$cement-mean(concrete$cement)
slag_c=concrete$slag-mean(concrete$slag)
fly_c=concrete$fly_ash-mean(concrete$fly_ash)
water_c=concrete$water-mean(concrete$water)
super_c=concrete$superplasticizer-mean(concrete$superplasticizer)
course_c=concrete$course_aggregate-mean(concrete$course_aggregate)
fine_c=concrete$fine_aggregate-mean(concrete$fine_aggregate)
age_c=concrete$age-mean(concrete$age)

# add squares and second-order interactions to data frame
concrete$cement_sq=cement_c*cement_c
concrete$slag_sq=slag_c*slag_c
concrete$fly_sq=fly_c*fly_c
concrete$water_sq=water_c*water_c
concrete$super_sq=super_c*super_c
concrete$course_sq=course_c*course_c
concrete$fine_sq=fine_c*fine_c
concrete$age_sq=age_c*age_c

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

concrete$fi_a_sq=fine_c*age_c


#########################################################
#                                                       #
#               PCR: Using OLS with PCs                 #
#                                                       #
#########################################################

# determine PCs for all predictors in concrete data set
pca_concrete=prcomp(concrete[,-4],scale=T)
summary(pca_concrete)

# view PCs and loadings
View(concrete[,-4])
View(pca_concrete$rot)
View(pca_concrete$x)

# perform OLS using PC1-PC
concrete_pcr=data.frame(cbind(pca_concrete$x,concrete$strength))
colnames(concrete_pcr)[43] <- "strength"

pcr.mod=lm(strength~.,data=concrete_pcr[,c(1:20,43)])
summary(pcr.mod)


#########################################################
#                                                       #
#      Select Best PC count using Cross-Validation      #
#                                                       #
#########################################################


k=5  # set number of folds
set.seed(123)
# create an index with id 1-5 to assign observations to folds
folds=sample(1:5,nrow(concrete_pcr),replace=T) 
# create dummy matrix to store CV error estimates
cv.err=matrix(NA,k,42,dimnames=list(NULL,paste(1:42)))

# perform CV
for (j in 1:k){
  # estimate test error for all 42 models by predicting kth fold 
  for (i in 1:42){
    lmod.cv=lm(strength~.,data=concrete_pcr[folds!=j,c(1:i,43)])
    pred=predict(lmod.cv,concrete_pcr[folds==j,])
    cv.err[j,i]=mean((concrete_pcr$strength[folds==j]-pred)^2)  # save error est
  }
}

mse.cv=apply(cv.err,2,mean) # cdompute mean MSE for each number of predictors
min=which.min(mse.cv)  # find minimum mean MSE

# plot and put a red circle around lowest MSE
par(mfrow=c(1,1))
plot(1:42,mse.cv,type="b",xlab="no. of predictors)",ylab="est. test MSE")
points(min,mse.cv[min],cex=2,col="red",lwd=2)
abline(h=seq(0,0.003,0.0005),lty=3)

# best MSE estimate is for 42, but 21 is close 
# fit best model (using all the data)

pcr.mod=lm(strength~.,data=concrete_pcr[,c(1:21,43)])
summary(pcr.mod)
par(mfrow=c(2,2))
plot(pcr.mod)
vif(pcr.mod)

pcr.mod=lm(strength~.,data=concrete_pcr[,c(1,3:21,43)])
summary(pcr.mod)
par(mfrow=c(2,2))
plot(pcr.mod)
vif(pcr.mod)

# compare to OLS
par(mfrow=c(1,1))
lmod=lm(strength~.,data=concrete)
plot(lmod$fitted.values,concrete$strength,pch=19,col="blue")
points(pcr.mod$fitted.values,concrete$strength,col="red",lwd=2)
abline(a=0,b=1)

# interpret?
View(pca_concrete$rot[,c(1,3:5)])

# fit with all nine PCs - OVERFIT
#pcr.mod=lm(strength~.,data=concrete_pcr)
#summary(pcr.mod)
#par(mfrow=c(2,2))
#plot(pcr.mod)
#vif(pcr.mod)





