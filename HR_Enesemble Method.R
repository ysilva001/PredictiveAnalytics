
HR <- read_csv("C:/Users/ysilva/Desktop/Data Science Practicum/HR_comma_sep.csv")

HR <- data.frame(HR)

##Change salary and Sales to numeric for correlation plot
HR$sales <- as.numeric(HR$sales)
HR$salary <- as.numeric(HR$salary)

sales <- unique(HR$sales)
HR$sales <- as.numeric(1:10)[match(HR$sales, sales)] 
HR$salary <- as.numeric(1:3)[match(HR$salary, c('low', 'medium', 'high'))]

#change classes of some of the objects so that numeric (0/1) values turn to factors including salary
HR$left <- as.factor(HR$left)
HR$promotion_last_5years<- as.factor(HR$promotion_last_5years)
HR$Work_accident <- as.factor(HR$Work_accident)
HR$salary <- as.factor(HR$salary)
HR$sales <- factor(HR$sales)
HR$salary <- ordered(HR$salary, c("low","medium" ,"high"))

#remove attributes no appropirate for classification features
HRTrain = data.frame(HR)
#Then, split 70 percent of the data into the training dataset and 30 percent of the data into the testing dataset
set.seed(2)
ind = sample(2, nrow(HRTrain), replace = TRUE, prob=c(0.7, 0.3))
trainset = HRTrain[ind == 1,]
testset = HRTrain[ind == 2,]
#adabag
set.seed(2)
HR.bagging = adabag::bagging(left ~ ., data=trainset, mfinal=10)
HR.bagging$mtrees
HR.predbagging= predict.bagging(HR.bagging, newdata=testset)
HR.predbagging$confusion 
HR.predbagging$error #0.0306

#cross validation using baggin.cv
HR.baggingcv = bagging.cv(left ~ ., v=10, data=trainset, mfinal=10)
HR.baggingcv$confusion
HR.baggingcv$error #.0604
#Boosting
set.seed(2)
HR.boost = boosting(left ~.,data=trainset,mfinal=10, coeflearn="Freund", boos=FALSE , control=rpart.control(maxdepth=3))
HR.boost.pred = predict.boosting(HR.boost,newdata=testset)
HR.boost.pred$confusion
HR.boost.pred$error #.058
plot(HR.boost$importance)
hist(HR.boost$importance)
print(HR.boost)

#Cross Validation with bosting.cv method
HR.boostcv = boosting.cv(left ~ ., v=10, data=trainset, mfinal=5,control=rpart.control(cp=0.01))
HR.boostcv$confusion
HR.boostcv$error #.0652

#Calculate Margins of a classifier
#change back to normal data
boost.margins = margins(HR.boost, trainset)
boost.pred.margins = margins(HR.boost.pred, testset)
plot(sort(boost.margins[[1]]), (1:length(boost.margins[[1]]))/length(boost.margins[[1]]), 
     type="l",xlim=c(-1,1),main="Boosting: Margin cumulative distribution graph", 
     xlab="margin", ylab="% observations", col = "blue")
lines(sort(boost.pred.margins[[1]]), (1:length(boost.pred.margins[[1]]))/length(boost.pred.margins[[1]]), type="l", col = "green")
abline(v=0, col="red",lty=2)

boosting.training.margin = table(boost.margins[[1]] > 0)
boosting.negative.training = as.numeric(boosting.training.margin[1]/boosting.training.margin[2])
boosting.negative.training

boosting.testing.margin = table(boost.pred.margins[[1]] > 0)
boosting.negative.testing = as.numeric(boosting.testing.margin[1]/boosting.testing.margin[2])
boosting.negative.testing

#Calculating Error Evolution
#Boosting
boosting.evol.train = errorevol(HR.boost, trainset)
boosting.evol.test = errorevol(HR.boost, testset)
plot(boosting.evol.test$error, type = "l", ylim = c(0, 1),
             main = "Boosting error versus number of trees", xlab = "Iterations",
             ylab = "Error", col = "red", lwd = 2)
lines(boosting.evol.train$error, cex = .5, col = "blue", lty = 2, lwd = 2)
legend("topright", c("test", "train"), col = c("red", "blue"), lty = 1:2, lwd = 2)

#RF
library(randomForest)
HR.rf = randomForest(left ~ ., data = trainset, importance = T)
HR.rf
HR.prediction = predict(HR.rf, testset)
table(HR.prediction, testset$left)
plot(HR.rf)
importance(HR.rf)
varImpPlot(HR.rf)
margins.rf=margin(HR.rf,trainset)
plot(margins.rf)
hist(margins.rf,main="Margins of Random Forest for HR dataset")
boxplot(margins.rf~trainset$left, main="Margins of Random Forest for HR dataset by class")

##RF MODEL - 200 trees
model_rf <- randomForest(as.factor(left) ~ ., data=trainset, nsize=3, ntree=500)
# Predict Output of test data
predicted_rf <- predict(model_rf, testset)
# Confusion matrix of random forest
table(testset$left, predicted_rf)
# Accuracy of random forest
mean(predicted_rf==testset$left)
#confusion matrix
print(confusionMatrix(predicted_rf, testset$left))

#party package rf model
library(party)
HR.cforest = cforest(left ~ ., data = trainset, controls=cforest_unbiased(ntree=1000, mtry=5))
HR.cforest
#make prediction from this
HR.cforest.prediction = predict(HR.cforest, testset, OOB=TRUE, type = "response")
table(HR.cforest.prediction, testset$HR)
#Estimating the prediction errors of different classifiers
HR.bagging= errorest(HR ~ ., data = trainset, model = bagging)
HR.bagging

library(ada)
HR.boosting= errorest(HR ~ ., data = trainset, model = ada)
HR.boosting
HR.rf= errorest(HR ~ ., data = trainset, model = randomForest)
HR.rf
#make a prediction function and use it to estimate the error rate of single decision tree
HR.predict = function(object, newdata) {predict(object, newdata = newdata, type = "class")}
HR.tree= errorest(HR ~ ., data = trainset, model = rpart,predict = HR.predict)
HR.tree
HR.boosting
