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
library(RColorBrewer) #Colors
library(randomForest) ##RF Algorithm 
library(randomForestSRC) ## Random Forests for Survival, Regression and Classification
library(caret) ## streamline the process for creating predictive models/feature selection/resampling
library(corrplot) ## graphical display of a correlation matrix
library(rpart) #decision trees
library(rpart.plot) #Plot an rpart model
library(pROC) #Display and Analyze ROC Curves
library(ROCR) #Performance measures
library(e1071) #svm/Naive Bayes
library(kernlab) #improve svm ksvm()
library(gridExtra) #grid arrange
library(psych) # how skewed each feature is
library(glm)





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
hr$sales <- factor(hr$sales)#, levels=c("management","hr","accounting",
#"RandD","marketing","product_mng",
#"IT","support","technical","sales"))


hr$sales <- as.character(hr$sales)
##COrrelation Plot
CorrData <- hr[,c("sales", "salary", "satisfaction_level","last_evaluation","number_project","promotion_last_5years","Work_accident", "average_montly_hours","time_spend_company","left")]
corrplot( cor(as.matrix(CorrData), method = "pearson"),is.corr = FALSE, 
          type = "lower", order = "hclust", tl.col = "black", tl.srt = 45)



#----------------ggplots------------------------#
#Satisfaction Level
p1 <- ggplot(hr, aes(x= satisfaction_level, fill=left)) + 
  geom_bar() + 
  scale_fill_brewer(palette="Spectral") +
  theme_classic() +
  labs(title="Satisfaction Level", x="Satisfaction Level", y="Count")
#Departments (Sales Variable)
levels(hr$sales)
p2 <- ggplot(hr, aes(x=sales, fill=left, geom="histogram"))+ 
  geom_bar() +
  coord_flip() +
  theme_classic() +
  ##theme(axis.text.x = element_text(angle = 30, vjust = 0.5)) +
  scale_fill_brewer(palette="Spectral") + 
  labs(title="Department", x="Position", y=" ") 


#Last Evaluation
p3 <- ggplot(hr, aes(x=last_evaluation, fill=left, geom="histogram"))+ 
  geom_bar() +
  scale_fill_brewer(palette="Spectral") + 
  theme_classic() +
  ggtitle("Last Evaluation")
#Number of Projects by Salary & By Left
p4 <- ggplot(hr, aes(x=salary, y=number_project, fill=left, geom="histogram"))+ 
  geom_bar(stat="identity") +
  scale_fill_brewer(palette="Spectral") + 
  theme_classic() +
  ggtitle("Number of Projects by Salary")+
  ylab(" ")
#Time spent at company & Employee Status
p5 <- ggplot(hr, aes(x=time_spend_company, fill=left, geom="histogram"))+ 
  geom_bar() +
  scale_fill_brewer(palette="Spectral") + 
  theme_classic() +
  ggtitle("Time spent at company") +
  ylab(" ")
#Number of projects & Salary & Employee Status
p6 <- ggplot(hr, aes(x =salary, colour = factor(number_project))) + 
  geom_density(alpha = 0.1) +
  facet_grid(. ~ left, scales = "free") + 
  scale_fill_brewer(palette="Spectral") +
  theme_classic() +
  ggtitle("Number of projects by Salary")
#Promotion in Last 5 Years
p7 <- ggplot(hr, aes(x=promotion_last_5years, fill=left)) + 
  geom_bar() +
  scale_fill_brewer(palette="Spectral") +
  theme_classic() +
  ggtitle("Promotion in Last 5 Years by Employee Status") +
  ylab(" ")
####promotion/ salary and promotion/left compare statistics
hr %>% 
  select(promotion_last_5years, salary) %>% 
  count(promotion_last_5years, salary) %>% 
  spread(promotion_last_5years, n)
sum(hr$promotion_last_5years== 1)/nrow(hr)
#Work Accident
p8 <- ggplot(hr, aes(x=Work_accident, fill=salary)) + 
  geom_bar() + 
  facet_grid(. ~ left, scales = "free") + 
  theme_classic() +
  scale_fill_brewer(palette="Spectral") +
  ggtitle("Work Accidents by Salary")
#Average Monthly Hours
p9 <- ggplot(hr, aes(x= average_montly_hours, fill=left)) + 
  geom_bar() +
  theme_classic() +
  scale_fill_brewer(palette="Spectral") +
  ggtitle("Average Monthly Hours")

grid.arrange(p6, p8, ncol = 2, nrow = 1)

#What is the correlation between satisfied and non satisfied workers and how mcuh do they earn?
unsatisfied_workers<- hr %>% 
  select(salary, satisfaction_level) %>% 
  filter(satisfaction_level <=0.6)

P10 <- ggplot(unsatisfied_workers, aes(x =  satisfaction_level, colour = factor(salary))) + 
  geom_density(size=2) + labs(title="Unsatisfied Workers and Salary", x="Satisfaction Level", y="Density")
P10 + scale_fill_brewer(palette="Spectral")

satisfied_workers<- hr %>% 
  select(salary, satisfaction_level) %>% 
  filter(satisfaction_level > 0.6)

P11 <- ggplot(satisfied_workers, aes(x =  satisfaction_level, colour = factor(salary))) + 
  geom_density(size=2) + labs(title="Satisfied Workers and Salary", x="Satisfaction Level", y="Density")
P11 + scale_fill_brewer(palette="Spectral")


#Last evaluation
summary(hr$last_evaluation)
unique(hr$last_evaluation)
##Does salary depend on last evaluation?
p12 <- ggplot(hr, aes(x =  last_evaluation, colour = factor(salary))) + 
  geom_density(size=1) + labs(title="Last Evaluation and Salary", x="Last Evaluation", y="Density") +
  scale_fill_brewer(palette="Spectral")