setwd("~/Documents/kaggle/bike_sharing")
train <- read.csv("train.csv", stringsAsFactors = FALSE, colClasses = c("datetime"="character"))
test <- read.csv("test.csv", stringsAsFactors = FALSE, colClasses = c("datetime"="character"))
# datetime - hourly date + timestamp  
# season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# holiday - whether the day is considered a holiday
# workingday - whether the day is neither a weekend nor holiday
# weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentals

# str(train)
# str(test)
# summary(train)

# creating a daytype variable:
# 1 = working day, 2 = holiday, 3 = weekend 
train$daytype <- 0
train$daytype[train$workingday == 1] <- 1
train$daytype[train$holiday == 1] <- 2
train$daytype[train$workingday == 0 & train$holiday == 0] <- 3

test$daytype <- 0
test$daytype[test$workingday == 1] <- 1
test$daytype[test$holiday == 1] <- 2
test$daytype[test$workingday == 0 & test$holiday == 0] <- 3
######################################
# create Hour variable
library(lubridate)
dates <- as.POSIXlt(train$datetime)
train$hour <- hour(dates)

datestest <- as.POSIXlt(test$datetime)
test$hour <- hour(datestest)
######################################
# create weekday variable
train$weekday_factor <- wday(dates, label = TRUE)
train$weekday <- as.integer(train$weekday_factor)

test$weekday_factor <- wday(datestest, label = TRUE)
test$weekday <- as.integer(test$weekday_factor)
######################################
# clean irrelevant variables
drops <- c("datetime","atemp","holiday","workingday","weekday","hour","daytype",
           "season","weather","count","casual","registered","count_factor")
predictors <- names(train)[! names(train) %in% drops]
# trainRelevant <- train[,!(names(train) %in% drops)]
# trainRelevant <- trainRelevant[,c(10,9,1,2,3,4,5,6,7,8)] # reorder variables
# head(trainRelevant)
# 
# testRelevant <- test[,!(names(test) %in% drops)]
# testRelevant <- testRelevant[,c(7,6,1,2,3,4,5)] # reorder variables
# head(testRelevant)
######################################
# factorize
str(train)
train$hour_factor <- factor(train$hour)
train$daytype_factor <- factor(train$daytype, 
                               labels = c("working day","holiday","weekend"))
train$season_factor <- factor(train$season,
                              labels = c("spring","summer","fall","winter"))
train$weather_factor <- factor(train$weather,
                               labels = c("Clear / partly cloudy","Mist","Light rain / snow", "Heavy rain / snow"))

train$count_factor <- 4
train$count_factor[train$count < 284] <- 3
train$count_factor[train$count < 145] <- 2
train$count_factor[train$count < 42] <- 1
train$count_factor <- factor(train$count_factor,
                             labels = c("low","moderate","high","extreme"))

test$hour_factor <- factor(test$hour)
test$daytype_factor <- factor(test$daytype,
                              labels = c("working day","holiday","weekend"))
test$season_factor <- factor(test$season,
                             labels = c("spring","summer","fall","winter"))
test$weather_factor <- factor(test$weather,
                              labels = c("Clear / partly cloudy","Mist","Light rain / snow", "Heavy rain / snow"))
######################################
# square rooting numerics
train$temp_sq <- sqrt(train$temp)
train$atemp_sq <- sqrt(train$atemp)
train$humidity_sq <- sqrt(train$humidity)
train$windspeed_sq <- sqrt(train$windspeed)

test$temp_sq <- sqrt(test$temp)
test$atemp_sq <- sqrt(test$atemp)
test$humidity_sq <- sqrt(test$humidity)
test$windspeed_sq <- sqrt(test$windspeed)
######################################
# Split data into train and test dataset
library(caret)
inTrain <- createDataPartition(y=train$count,p=0.7,list=FALSE) 
training <- train[inTrain,]
testing <- train[-inTrain,]
dim(training);dim(testing)
summary(testing$count)
str(training)
dim(training)
dim(testing)
######################################
# Regression
# check correlations
# trainingCheck <- training[,-c(1,15,17,18,19,20)]
# head(trainingCheck)
# 
# library(psych)
# pairs.panels(trainingCheck,lm=TRUE)

###################################
# setting predictors
formulaCount <- count ~
  temp + hour_factor + season_factor + humidity + daytype_factor
formulaReg <- registered ~ 
  temp + hour_factor + season_factor + humidity + daytype_factor
formulaCas <- casual ~ 
  temp + hour_factor + season_factor + humidity + daytype_factor

# another way
extractFeatures <- function(data) {
  features <- c("temp_sq",
                "season",
                "hour",
                "daytype",
                "weather",
                "humidity_sq",
                "windspeed_sq")
  return(data[,features])
}
##############################
# identify high correlations
library(mlbench)
# calculate correlation matrix
data <- extractFeatures(training)
correlationMatrix <- cor(data)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)
head(extractFeatures(training))

##############################
# automatic feature selection
# library(plyr); library(dplyr)
# # define the control using a random forest selection function
# control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# # run the RFE algorithm
# results <- rfe(data, training$count, sizes=c(1:8), rfeControl=control)
# # summarize the results
# print(results)
# # list the chosen features
# predictors(results)
# # plot the results
# plot(results, type=c("g", "o"))
# 
# summary(lm(count~hour_factor,data=training))$adj.r.squared #.5170446
# summary(lm(count~daytype_factor,data=training))$adj.r.squared #-0.0002481019
# summary(lm(count~season_factor,data=training))$adj.r.squared #0.06030486
# summary(lm(count~weather_factor,data=training))$adj.r.squared #0.01638079
# summary(lm(count~temp,data=training))$adj.r.squared #0.1477739
# summary(lm(count~humidity,data=training))$adj.r.squared #0.09577633
# summary(lm(count~windspeed,data=training))$adj.r.squared #0.01099768
# fit1 <- lm(count~hour_factor,data=training)
# summary(fit1)
# 
# summary(lm(count~hour_factor + temp,data=training))$adj.r.squared #0.5912693
# summary(lm(count~hour_factor + humidity,data=training))$adj.r.squared #0.521842
# summary(lm(count~hour_factor + season_factor,data=training))$adj.r.squared #0.5805221
# summary(lm(count~hour_factor + windspeed,data=training))$adj.r.squared #0.5197088
# summary(lm(count~hour_factor + daytype_factor,data=training))$adj.r.squared #0.5169322
# 
# summary(lm(count~hour_factor + temp + season_factor,data=training))$adj.r.squared #0.6070909
# summary(lm(count~hour_factor + temp + humidity,data=training))$adj.r.squared #0.5990329
# 
# #######
# # svm
# library(e1071)
# LMcount <- svm(formulaCount,training)
# predCount <- predict(LMcount,testing)
# qplot(count,predCount,data=testing)
# 
# # checking root mean squared error
# rmse <- function(error)
# {
#   sqrt(mean(error^2))
# }
# error <- testing$count - predCount
# svrPredictionRMSE <- rmse(error)  # 92.65736
# 
# 
# LMreg <- svm(formulaReg,training)
# predReg <- predict(LMreg,testing)
# qplot(registered,predReg,data=testing)
# # checking root mean squared error
# 
# errorReg <- testing$registered - predReg
# rmseReg <- rmse(errorReg)  # 81.26671
# 
# 
# LMcas <- svm(formulaCas,training)
# predCas <- predict(LMcas,testing)
# qplot(casual,predCas,data=testing)
# qplot(temp,casual,data=testing)
# points(testing$temp,predCas, col = "red", pch=4)
# # checking root mean squared error
# errorCas <- testing$casual - predCas
# rmseCas <- rmse(errorCas)  # 26.4457
# head(testing$casual)
# head(predCas)
# 
# ModFitReg <- LMreg
# ModFitCas <- LMcas
######################################
# Random Forests
library(randomForest)

# using train function
ModFitReg <- train(y=training$registered, x=extractFeatures(training), data = training, method = "rf", do.trace=T, ntree = 100,
                trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))

ModFitCas <- train(y=training$casual, x=extractFeatures(training), data = training, method = "rf", do.trace=T, ntree = 100,
                   trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))
print(ModFitCas)



ModFitCount <- train(y=training$count, x=extractFeatures(training), data = training, method = "rf", do.trace=T, ntree = 100, importance=TRUE,
                   trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))
print(ModFitCount)

######################################
# boosting

# ModFitReg <- train(registered ~ 
#                   temp + hour_factor + season + humidity + daytype,
#                 method = "gbm", data = training, verbose = FALSE)
# 
# ModFitCas <- train(casual ~ 
#                      temp + hour_factor + season + humidity + daytype,
#                    method = "gbm", data = training, verbose = FALSE)
# 
# 

###############################################################
# benchmarking
# registered + casual
testing$predReg <- predict(ModFitReg, newdata=testing)
testing$predCas <- predict(ModFitCas, newdata=testing)
testing$prediction <- testing$predReg + testing$predCas

# count only
testing$prediction <- predict(ModFitCount, newdata=testing)
#confusionMatrix(data=testing$Prediction, testing$count) #inefficient with continuous variables 

# using RMSLE
library('Metrics')
rmsle(testing$count, testing$prediction)

###########################################
# predicting on test data
test$predReg <- as.integer(predict(ModFitReg, newdata=test))
test$predCas <- as.integer(predict(ModFitCas, newdata=test))
test$prediction <- test$predReg + test$predCas
head(test)
head(train)
submission <- data.frame(datetime = test$datetime, count = test$prediction)
head(submission)
write.csv(submission, "submission21_5_b.csv", row.names=FALSE)
# using randomForest function
forest_model <- randomForest(y=training$count, x=training[,-(8:10)], 
                             ntree=100,do.trace=T)
pred <- predict(forest_model,testing)
pred <- as.integer(pred)
testing$predRight <- pred == testing$count
table(pred,testing$count)

qplot(hour,temp,colour=predRight,data=testing)





