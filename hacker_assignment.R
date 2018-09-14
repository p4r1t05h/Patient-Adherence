rm(list=ls())

#setting working directory
setwd("D:/Data Science/Abzooba")
getwd()

#Loading important libraries
library(rpart)
library(C50)
library(randomForest)
library(class)
library(e1071)
library(caret)
library(boot)

#Loading data sets 
train<-read.csv("Training Data.csv", header = T)
test<-read.csv("Test Data.csv", header = T)

str(train)
head(train)

#checking for missing data
table(is.na(train))

#As we can see there are no missing data

#Now we will convert the variables into proper data type

train$Diabetes<-as.factor(train$Diabetes)
test$Diabetes<-as.factor(test$Diabetes)
train$Alcoholism<-as.factor(train$Alcoholism)
test$Alcoholism<-as.factor(test$Alcoholism)
train$HyperTension<-as.factor(train$HyperTension)
test$HyperTension<-as.factor(test$HyperTension)
train$Smokes<-as.factor(train$Smokes)
test$Smokes<-as.factor(test$Smokes)
train$Tuberculosis<-as.factor(train$Tuberculosis)
train$Sms_Reminder<-as.factor(train$Sms_Reminder)

str(train)
str(test)

#====Univariate Analysis
#Quantitative Variables

#Numerical analysis of Age | 5 point summary

age.5p<-summary(train$Age)
age.5p

#Visual analysis for Age | Histogram and Boxplot

hist(train$Age, main = "Histogram of Age")
boxplot(train$Age, names = "Boxplot of Age")


#Numerical analysis of Prescription Period | 5 point summary

pres.5p<-summary(train$Prescription_period)
pres.5p

#Visual analysis for Prescription Period | Histogram and Boxplot

hist(train$Prescription_period, main = "Histogram of Prescription Period")
boxplot(train$Prescription_period, names = "Boxplot of Prescription PEriod")

#===Bivariate Analysis
#Numerical Analysis for Gender | Proportion Table

prop.table(table(train$Gender))*100
gen.prop<-as.matrix(table(train$Gender))

#Visual Analysis for Gender | Bar Graph & Pie chart

barplot(gen.prop, col = c(3,4), horiz = FALSE)
pie(gen.prop, labels = c("1=F", "2=M"))

#Numerical Analysis for Diabetes | Proportion Table

prop.table(table(train$Diabetes))*100
diabetes.prop<-as.matrix(table(train$Diabetes))

#Visual Analysis for Gender | Bar Graph & Pie chart

barplot(diabetes.prop, col = c(3,4), horiz = FALSE)
pie(diabetes.prop, labels = c("1=No", "2=Yes"))

#Numerical Analysis for Alchoholism | Proportion Table

prop.table(table(train$Alcoholism))*100
alchohol.prop<-as.matrix(table(train$Alcoholism))

#Visual Analysis for Gender | Bar Graph & Pie chart

barplot(alchohol.prop, col = c(3,4), horiz = FALSE)
pie(alchohol.prop, labels = c("1=No", "2=Yes"))

#Numerical Analysis for Hypertension | Proportion Table

prop.table(table(train$HyperTension))*100
hyper.prop<-as.matrix(table(train$HyperTension))

#Visual Analysis for Gender | Bar Graph & Pie chart

barplot(hyper.prop, col = c(3,4), horiz = FALSE)
pie(hyper.prop, labels = c("1=No", "2=Yes"))

#Numerical Analysis for Smokes | Proportion Table

prop.table(table(train$Smokes))*100
smoke.prop<-as.matrix(table(train$Smokes))

#Visual Analysis for Gender | Bar Graph & Pie chart

barplot(smoke.prop, col = c(3,4), horiz = FALSE)
pie(smoke.prop, labels = c("1=No", "2=Yes"))

#Numerical Analysis for Tuberculosis | Proportion Table

prop.table(table(train$Tuberculosis))*100
tuber.prop<-as.matrix(table(train$Tuberculosis))

#Visual Analysis for Gender | Bar Graph & Pie chart

barplot(tuber.prop, col = c(3,4), horiz = FALSE)
pie(tuber.prop, labels = c("1=No", "2=Yes"))

#Numerical Analysis for SMS Reminder | Proportion Table

prop.table(table(train$Sms_Reminder))*100
sms.prop<-as.matrix(table(train$Sms_Reminder))

#Visual Analysis for Gender | Bar Graph & Pie chart

barplot(sms.prop, col = c(3,4), horiz = FALSE)
pie(sms.prop, labels = c("1=0 Reminder", "2=1 Reminder", "3=2 Reminders"))


#====Bivariate Analysis and Dependency Check
#Y=Adherence (Y is categorical Variable)

#Numerical Analysis Gender->Adherence | 2 Way Table

gen.table<-table(train$Gender,train$Adherence)
gen.table

#Chi Squared test for Dependency
gen.chi<-chisq.test(gen.table)
gen.chi

#since p<0.05 | It is DEPENDET

#Numerical Analysis Diabetes->Adherence | 2 Way Table

dia.table<-table(train$Diabetes,train$Adherence)
dia.table

#Chi Squared test for Dependency
dia.chi<-chisq.test(dia.table)
dia.chi

#since p<0.05 | It is DEPENDET

#Numerical Analysis Alcoholism->Adherence | 2 Way Table

al.table<-table(train$Alcoholism,train$Adherence)
al.table

#Chi Squared test for Dependency
al.chi<-chisq.test(al.table)
al.chi

#since p<0.05 | It is DEPENDET

#Numerical Analysis Hypertension->Adherence | 2 Way Table

hyper.table<-table(train$HyperTension,train$Adherence)
hyper.table

#Chi Squared test for Dependency
hyper.chi<-chisq.test(hyper.table)
hyper.chi

#since p<0.05 | It is DEPENDET

#Numerical Analysis Gender->Adherence | 2 Way Table

smokes.table<-table(train$Smokes,train$Adherence)
smokes.table

#Chi Squared test for Dependency
smokes.chi<-chisq.test(smokes.table)
smokes.chi

#since p<0.05 | It is DEPENDET

#Numerical Analysis Tuberculosis->Adherence | 2 Way Table

tuber.table<-table(train$Tuberculosis,train$Adherence)
tuber.table

#Chi Squared test for Dependency
tuber.chi<-chisq.test(tuber.table)
tuber.chi

#since p>0.05 | It is InDEPENDET

#Numerical Analysis SMS Reminder->Adherence | 2 Way Table

sms.table<-table(train$Sms_Reminder,train$Adherence)
sms.table

#Chi Squared test for Dependency
sms.chi<-chisq.test(sms.table)
sms.chi

#since p<0.05 | It is InDEPENDET

#Removing the independent variables from data set

train<-train[,c(-9,-10)]
test<-test[,c(-9,-10)]

#=========================Model Building===================================

#we are splitting the training data to Validation set data because
#we cannot build Confusion Matrix without the Target Variable which is absent from the TEST DATA
#Splitting the traing set into training and Validation set

set.seed(1)
val.index = createDataPartition(train$Adherence, p = .80, list = FALSE)
train = train[ val.index,]
validation  = train[-val.index,]

#Since we have to calculate Probablity score for each patient
#We'll only build Logistic Regression Model

#============================================
#======Logistic Regression Model
#============================================

model1<-glm(train$Adherence~., data = train, family = "binomial")

coef(model1)
summary(model1)$coef
summary(model1)

#Predicting using Logistic regression model

predict.model1<-predict.glm(model1, newdata = validation[,-9], type = "response")

#Converting Probablities into 0 & 1

prob.model1<-ifelse(predict.model1>0.5,1,0)

#Creating Confusion Matrix For Logistic Regression

confusion.model1<-table(Predicted=prob.model1,Actual=validation$Adherence)
confusion.model1

#Determining the Accuracy and Error Rate

sum(confusion.model1[c(1,4)]/sum(confusion.model1[1:4])) #Correct Prediction
1-sum(confusion.model1[c(1,4)]/sum(confusion.model1[1:4])) #Prediction Error

#Precision for Yes= 7173/(7173+1894)=79.11%
#Recall for Yes=7173/(7173+1496)=82.74%
#Precision for No = 18180/(18180+1496)=92.4%
#Recall for No = 18180/(18180+1894)=90.56%


#Building LOGISTIC REGRESSION MODEL USING TEST DATA

LRmodel<-glm(train$Adherence~., data = train, family = "binomial")

summary(LRmodel)

#Predicting using Logistic regression model

predict.LRmodel<-predict.glm(LRmodel, newdata = test, type = "response")

#Since there is no specification given as to how much probablity score will result in Adherence
#then we'll assume that if the probablity is more than 0.5 than the Prediction is YES if not then the Prediction is No


predict.LR<-ifelse(predict.LRmodel>0.5,"Yes","No")

final.results<-cbind(predict.LR,predict.LRmodel)

head(final.results)

colnames(final.results)<-c("Probablity Score", "Adherence")

write.csv(final.results,"./Final.csv")

### I have made other Machine Learning Models just in case

#==============================================
#====Decision tree Model
#==============================================

#model2 = C5.0(train$Adherence ~., data=train, trials = 100, rules = TRUE)

#Summary of DT model
#summary(model2)

#Predicting for test cases
#predict.model2 = predict(model2, newdata=validation[,-9], type = "class")

#Creating Confusion Matrix for Decision Tree

#confusion.model2<-table(validation$Adherence,predict.model2)
#confusionMatrix(confusion.model2)

#Determining the Accuracy and Error Rate
#sum(confusion.model2[c(1,4)]/sum(confusion.model2[1:4])) #Correct Prediction
#1-sum(confusion.model2[c(1,4)]/sum(confusion.model2[1:4])) #Prediction Error

#Precision for Yes= 7562/(8205+1946)=74.49%
#Recall for Yes= 7562/(7562+1107)= 87.23%
#Precision for No = 18128/(18128+1107)=94.24%
#Recall for No = 18128/(18128+1946)=90.30%

#=====Random Forest Model
#model3<-randomForest(train$Adherence~., data = train, importance=T, ntree=500)

#summary(model3)

#predict.model3<-predict(model3, validation[,-9])

#confusion.model3<-table(Predicted=prob.model3,Actual=validation$Adherence)
#confusionMatrix(confusion.model3)

#Determining the Accuracy and Error Rate
#sum(confusion.model3[c(1,4)]/sum(confusion.model3[1:4])) #Correct Prediction
#1-sum(confusion.model3[c(1,4)]/sum(confusion.model3[1:4])) #Prediction Error

#Precision for Yes = 7772/(7772+2093)=78.78%
#Recall for Yes= 7772/(7772+897)=89.65%
#Precision for No = 17981/(17981+897)=95.24%
#Recall for No = 17981/(17981+2093)=89.57%

#===========================
#=====KNN Model
#===========================
#model4<-knn(train[,1:8], validation[,1:8], train$Adherence, k=7)

#==========================
#====Naive Bayes Model
#==========================

#model5<-naiveBayes(train$Adherence~., data = train)

#Predicting Model

#predict.model5<-predict(model5, newdata = validation[,-9], type = "class")

#Confusion Matrix for Naive Bayes

#confusion.model5<-table(Predicted=prob.model5,Actual=validation$Adherence)
#confusionMatrix(confusion.model5)

#Determining the Accuracy and Error Rate
#sum(confusion.model5[c(1,4)]/sum(confusion.model5[1:4])) #Correct Prediction
#1-sum(confusion.model5[c(1,4)]/sum(confusion.model5[1:4])) #Prediction Error

#Precision for Yes= 7620/(7620+2840)=72.84%
#Recall for Yes= 7620/(7620+1049)=87.89% 
#Precision for No = 17234/(17234+1049)=94.26%
#Recall for No = 17234/(17234+2840)=85.85%

