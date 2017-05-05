###################
# Title : Human Resources Analytics
# Data Source: Kaggle 
# Author : Yesenia Silva
# MSDS696x70_Data Science Practicum II
###################

##############Packages#############
library(readr)  ##Read the file
library(dplyr) ##Data Manipulation
library(tidyr) ##Data Manipulation used in conjunction with dplyr
library(ggplot2) ##Visuals
library(ggvis) ##create applications for interactive data exploration and presentation
library(corrplot)  ##graphical display of a correlation matrix, confidence interval
library(DT) ##provides an R interface to the JavaScript library DataTables
library(magrittr) ## %>%
library(RColorBrewer) #Colors
library(randomForest) ##RF Algorithm 
library(randomForestSRC) ## Random Forests for Survival, Regression and Classification
library(caret) ## streamline the process for creating predictive models/feature selection/resampling
library(corrplot) ## graphical display of a correlation matrix
library(rattle) ##utility functions and the graphical user interface
library(xgboost)##Boost Training
library(rpart) #decision trees
library(rpart.plot) #Plot an rpart model
library(pROC) #Display and Analyze ROC Curves
library(ROCR) #Performance measures
library(e1071) #svm/Naive Bayes
library(kernlab) #improve svm ksvm()
library(gridExtra) #grid arrange
library(psych) # how skewed each feature is
library(mboost) #Boosting Method
library(ada)




###########Load Data###############
setwd("C:/Users/ysilva/Desktop/Data Science Practicum")
hr <- read_csv("C:/Users/ysilva/Desktop/Data Science Practicum/HR_comma_sep.csv")

###########Data Exploration###############
glimpse(hr)
names(hr)
sapply(hr, function(x) length(unique(x))) ##Check for unique values for each variable
unique(hr$salary)
head(hr)
summary(hr)
str(hr)
sapply(hr,class)
hr <- data.frame(hr)

##Change salary and Sales to numeric for correlation plot
hr$sales <- as.numeric(hr$sales)
hr$salary <- as.numeric(hr$salary)

sales <- unique(hr$sales)
hr$sales <- as.numeric(1:10)[match(hr$sales, sales)] 
hr$salary <- as.numeric(1:3)[match(hr$salary, c('low', 'medium', 'high'))]

#change classes of some of the objects so that numeric (0/1) values turn to factors including salary
hr$left <- as.factor(hr$left)
hr$promotion_last_5years<- as.factor(hr$promotion_last_5years)
hr$Work_accident <- as.factor(hr$Work_accident)
hr$salary <- as.factor(hr$salary)
hr$sales <- factor(hr$sales)
hr$salary <- ordered(hr$salary, c("low","medium" ,"high"))



#-------------------Predictive Models-----------------------------#
#how skewed each feature is and stand deviation
hr <- read_csv("C:/Users/ysilva/Desktop/Data Science Practicum/HR_comma_sep.csv")
hr$sales <- as.numeric(hr$sales)
hr$salary <- as.numeric(hr$salary)
describe(hr)

#-------------------Important Features-----------------------------#
#Reload data at this point to revert factors back

## Partition the data set into training set and test set with 80:20 ratios 
set.seed(2)
ind <- sample(2, nrow(hr), replace = TRUE, prob=c(0.8, 0.2))
train = hr[ind == 1,]
test = hr[ind == 2,]

dim(train)
dim(test)
str(train)
###################### Logistic Regression ######################
#78.59% accuracy
# Train the model using the training sets and check score
model_glm <- glm(left ~ ., data = train, family='binomial')
# Predict Output of test data
predicted_glm <- predict(model_glm, test, type='response')
predicted_glm <- ifelse(predicted_glm > 0.5,1,0)
# Confusion matrix of Logistic regression
table(test$left, predicted_glm)
# Accuracy of model
mean(predicted_glm==test$left)
###################### Decision Tree ############################
#96.66% accuracy 
model_dt <- rpart(left ~ ., data=train, method="class", minbucket=25) # class = classification tree
rpart.plot(model_dt)
#Variable Importance
model_dt_vi <- model_dt$variable.importance
plot(model_dt_vi)
# View decision tree plot
rpart.plot(model_dt)
# Predict Output of test data
predicted_dt <- predict(model_dt, test, type="class") 
# Confusion matrix of decision tree
table(test$left, predicted_dt)
# Accuracy of decision tree
mean(predicted_dt==test$left)
# Print results
print(model_dt)
#display cp table
printcp(model_dt) #Shows variables used in tree construction
#plot cross-validation resutls
plotcp(model_dt)
###################### SVM ######################################
# using simple linear kernel function 94.64%
#turn applicable variables to factors 
model_svm <- svm(left ~ ., data=train)
# Predict Output of test data
predicted_svm <- predict(model_svm, test)
predicted_svm <- ifelse(predicted_svm > 0.5,1,0)
# compare the predicted leftr to the true left in the testing dataset
table(predicted_svm, test$left)
# T/F Vector to see if model matches test data
agreement <- predicted_svm == test$left
table(agreement)
prop.table(table(agreement))
# Confusion matrix of SVM
table(test$left, predicted_svm)
# Accuracy of SVM
mean(predicted_svm==test$left)
###################### RF #######################################
#98.77%
#turn characters to factors 
hr$left <- as.factor(hr$left)
hr$promotion_last_5years<- as.factor(hr$promotion_last_5years)
hr$Work_accident <- as.factor(hr$Work_accident)
hr$salary <- ordered(hr$salary, c("low","medium" ,"high"))
hr$sales <- as.factor(hr$sales)
#then re run test/train datasets
model_rf <- randomForest(as.factor(left) ~ ., data=train, nsize=20, ntree=200)
# Predict Output of test data
predicted_rf <- predict(model_rf, test)
# Confusion matrix of random forest
table(test$left, predicted_rf)
# Accuracy of random forest
mean(predicted_rf==test$left)
#confusion matrix
print(confusionMatrix(predicted_rf, test$left))

###################### BOOSTING w/ adaboost package ######################
#predict the model 
hr.boost.pred <- predict.boosting(hr.boost,newdata=test)
#confusion matrix
hr.boost.pred$confusion
#Accuracy
(100 - (hr.boost.pred$error))
#Average Error from predicted results
hr.boost.pred$error

#10-fold cross-validate the training data using boosting on v-10 subsets
hr.boostcv <- boosting.cv(left ~ ., v=10, data=train, mfinal=5,control=rpart.control(cp=0.01))
#obtain confusion matrix
hr.boostcv$confusion
#Accuracy 
(100 - (hr.boostcv$error))
#Error
hr.boostcv$error
#caclculate the margin of boosting ensemble learner
boost.margins <- margins(hr.boost, train)
boost.pred.margins <- margins(hr.boost.pred, test)
#plot a marginal cumulative distribution graph of the boosting classifiers
plot(sort(boost.margins[[1]]), 
     (1:length(boost.margins[[1]]))/length(boost.margins[[1]]), 
     type="l",xlim=c(-1,1),main="Boosting: Margin cumulative distribution graph", 
     xlab="margin", ylab="% observations", 
     col = "blue")
lines(sort(boost.pred.margins[[1]]), 
      (1:length(boost.pred.margins[[1]]))/length(boost.pred.margins[[1]]), 
      type="l", col = "green")
abline(v=0, col="red",lty=2)
#calculate the percentage of negative margin matches training errors and the percentage of negative margin matches test errors
boosting.training.margin = table(boost.margins[[1]] > 0)
boosting.negative.training = as.numeric(boosting.training.margin[1]/boosting.training.margin[2])
boosting.negative.training

boosting.testing.margin = table(boost.pred.margins[[1]] > 0)
boosting.negative.testing = as.numeric(boosting.testing.margin[1]/boosting.testing.margin[2])
boosting.negative.testing

#calculate the error evolution of the bossting classifier
boosting.evol.train = errorevol(hr.boost, train)
boosting.evol.test = errorevol(hr.boost, test)
plot(boosting.evol.test$error, type = "l", ylim = c(0, 1),
     main = "Boosting error versus number of trees", xlab = "Iterations",
     ylab = "Error", col = "red", lwd = 2)
lines(boosting.evol.train$error, cex = .5, col = "blue", lty = 2, lwd = 2)
legend("topright", c("test", "train"), col = c("red", "blue"), lty = 1:2, lwd = 2)


###################### BOOSTING w/ mboost package ######################
set.seed(2)
ctrl <- trainControl(method = "repeatedcv", repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary)
ada.train <- train(left ~ ., data = train, method = "ada", metric = "ROC", trControl = ctrl)
  
  
######################PLOT THE ROC CURVES FOR ALL THE MODELS##############################
# Logistic regression
predict_glm_ROC <- predict(model_glm, test, type="response")
pred_glm <- prediction(predict_glm_ROC, test$left)
perf_glm <- performance(pred_glm, "tpr", "fpr")

# Decision tree
predict_dt_ROC <- predict(model_dt, test)
pred_dt <- prediction(predict_dt_ROC[,2], test$left)
perf_dt <- performance(pred_dt, "tpr", "fpr")

# Random forest
predict_rf_ROC <- predict(model_rf, test, type="prob")
pred_rf <- prediction(predict_rf_ROC[,2], test$left)
perf_rf <- performance(pred_rf, "tpr", "fpr")

# SVM
predict_svm_ROC <- predict(model_svm, test, type="response")
pred_svm <- prediction(predict_svm_ROC, test$left)
perf_svm <- performance(pred_svm, "tpr", "fpr")


# Area under the ROC curves
auc_glm <- performance(pred_glm,"auc")
auc_glm <- round(as.numeric(auc_glm@y.values),3)
auc_dt <- performance(pred_dt,"auc")
auc_dt <- round(as.numeric(auc_dt@y.values),3)
auc_rf <- performance(pred_rf,"auc")
auc_rf <- round(as.numeric(auc_rf@y.values),3)
auc_svm <- performance(pred_svm,"auc")
auc_svm <- round(as.numeric(auc_svm@y.values),3)
print(paste('AUC of Logistic Regression:',auc_glm))
print(paste('AUC of Decision Tree:',auc_dt))
print(paste('AUC of Random Forest:',auc_rf))
print(paste('AUC of Support Vector Machine:',auc_svm))


# Plotting the three curves
plot(perf_glm, main = "ROC curves for the models", col='blue')
plot(perf_dt,add=TRUE, col='red')
plot(perf_rf, add=TRUE, col='green3')
plot(perf_svm, add=TRUE, col='darkmagenta')
legend('bottom', c("Logistic Regression", 
                   "Decision Tree", "Random Forest", 
                   "Support Vector Machine"), 
       fill = c('blue','red','green3','darkmagenta'), 
       bty='n')

importanceplot(hr.boost,horiz=FALSE)
,element_text(angle = 30))
               horiz=TRUE, cex.names = .8)element_text(angle = 30