#=================================================================
#load libraries 
#=================================================================
library(readr) 
library(mlbench)
library(caret)
library(tidyverse)
library(ggplot2)
library(doSNOW)
library(gbm)
library(C50)
library(fastDummies)
library(corrplot)
library(GGally)
library(psych)
library(reshape)
library(e1071)
library(rminer)
library(gtsummary)
library(lubridate)
library(reshape2)
library(superml)
library(dplyr)
library(ggmap)
library(doParallel)
library(rpart)
library(rpart.plot)
library(data.tree)
library(caTools)
library(plotly)
library(pROC)
library(mlbench)
library(kknn)

#=================================================================
#parallel processing on
#=================================================================
# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(3)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 


#=================================================================
#load data iphone small matrix
#=================================================================
a <- file.choose()
df_ip <-read.csv(a)

#Please use the following to understand labeled sentiment
#0: Sentiment Unclear
#1: very negative
#2: somewhat negative
#3: neutral
#4: somewhat positive
#5: very positive


summary(df_ip) 
str(df_ip)
names(df_ip)
sum(is.na(df_ip)) #no na 
duplicated(df_ip)
df_ip[duplicated(df_ip),]
sum(duplicated(df_ip))
#=================================================================
#finding correlation 
#=================================================================
correlationMatrix <- cor(df_ip)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)



corrData <- cor(df_ip)
corrData
corrplot(corrData)

#What classes of data are the attributes and y-variable?
class(df_ip)
typeof(df_ip)
str(df_ip) 

#integer and list 
#=================================================================
#What is the distribution of the dependent variable? 
#=================================================================
plot_ly(df_ip, x= ~df_ip$iphonesentiment, type='histogram')
# 5 is more and next is 0 

#Pre-processing & Feature Selection

#=================================================================
# create a new data set and remove features highly correlated with the dependent 
#=================================================================
#Examine Correlation (Do Classification problems suffer from Collinearity?)

iphoneCOR <- df_ip
iphoneCOR$iphone <- NULL
iphoneCOR$nokialumina <- NULL
iphoneCOR$htcphone <- NULL
iphoneCOR$ios<- NULL
iphoneCOR$iphonecampos <-NULL
iphoneCOR$nokiacampos <-NULL
iphoneCOR$sonycamneg<- NULL
iphoneCOR$nokiacamneg<- NULL
iphoneCOR$sonycamunc<- NULL
iphoneCOR$nokiacamunc<- NULL
iphoneCOR$iphonecamunc<- NULL
iphoneCOR$iphonedispos <- NULL
iphoneCOR$sonydispos <- NULL 
iphoneCOR$nokiadispos<- NULL
iphoneCOR$iphonedisneg <- NULL 
iphoneCOR$sonydisneg <- NULL 
iphoneCOR$nokiadisneg <- NULL
iphoneCOR$htcdispos<- NULL
iphoneCOR$nokiadisneg<- NULL
iphoneCOR$sonydisunc<- NULL
iphoneCOR$nokiadisunc<- NULL
iphoneCOR$iphoneperneg<- NULL
iphoneCOR$sonyperunc<- NULL
iphoneCOR$nokiaperunc<- NULL
iphoneCOR$iosperpos<- NULL
iphoneCOR$iosperneg<- NULL
iphoneCOR$iosperunc<- NULL
iphoneCOR$iphoneperpos<- NULL
iphoneCOR$sonyperpos<- NULL
iphoneCOR$nokiaperpos<- NULL
iphoneCOR$sonyperneg<- NULL
iphoneCOR$nokiaperneg<- NULL


iphonecorrData <- cor(iphoneCOR)
iphonecorrData
corrplot(iphonecorrData)
#=================================================================
#Examine Feature Variance
#=================================================================
#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 
nzvMetrics <- nearZeroVar(df_ip, saveMetrics = TRUE)
nzvMetrics
# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(df_ip, saveMetrics = FALSE) 
nzv
#Does your "nzv" object align with your "nzvMetrics" results?  Yes, it does.
# create a new data set and remove near zero variance features
iphoneNZV <- df_ip[,-nzv]
iphone.nzv <- df_ip[,-nzv]
iphone.nzv$iphonesentiment <- as.numeric(iphone.nzv$iphonesentiment)
str(iphone.nzv)
iphoneNZVData <- cor(iphone.nzv)
iphoneNZVData
corrplot(iphoneNZVData)
#=================================================================
#Recursive Feature Elimination 
#=================================================================
# Let's sample the data before using RFE
#set.seed(123)
#iphoneSample <- df_ip[sample(1:nrow(df_ip), 1000, replace=FALSE),]
#View(iphoneSample)
# Set up rfeControl with randomforest, repeated cross validation and no updates
#ctrl <- rfeControl(functions = rfFuncs, 
     #              method = "repeatedcv",
       #            repeats = 3,
       #            verbose = FALSE)
# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
#rfeResults <- rfe(iphoneSample[,1:58], 
   #               iphoneSample$iphonesentiment, 
   #               sizes=(1:58), 
   #               rfeControl=ctrl)
# Get results
#rfeResults


# Define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 5, # number of repeats
                      number = 10) # number of folds
set.seed(2021)
inTrain <- createDataPartition(df_ip$iphonesentiment, p = .10, list = FALSE)
View(inTrain)
x_train <- df_ip[inTrain, ]
x_test  <- df_ip[-inTrain, ]
View(x_test[,1:58])
y <- df_ip$iphonesentiment
y_train <- y[inTrain]
y_test  <- y[-inTrain]
# Run RFE
result_rfe1 <- rfe(x = x_train[,1:58], 
                   y = y_train, 
                   sizes = c(1:58),
                   rfeControl = control)

# Print the results
result_rfe1

# Print the selected features
predictors(result_rfe1)



# Plot results
plot(result_rfe1, type=c("g", "o"))

# create new data set with rfe recommended features
iphoneRFE <- df_ip[,predictors(result_rfe1)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- df_ip$iphonesentiment

# review outcome
str(iphoneRFE)
iphone.rfe <- iphoneRFE
iphone.rfe$iphonesentiment <- as.numeric(iphone.rfe$iphonesentiment)
iphoneRFEData <- cor(iphone.rfe)
iphoneRFEData
corrplot(iphoneRFEData) 

summary(df_ip)
summary(iphoneCOR)
summary(iphoneNZV)
summary(iphoneRFE)


meltData <- melt(df_ip)
p <- ggplot(meltData, aes(factor(variable), value))
p + geom_boxplot() + facet_wrap(~variable, scale="free")

#=================================================================
#Rank Feature by importance 
#=================================================================
set.seed(7)
# prepare training scheme
control1 <- trainControl(method="repeatedcv", number=3, repeats=2)
# train the model
model1 <- train(iphonesentiment~., data=df_ip, method="rf", preProcess="scale", trControl=control1)
# estimate variable importance
importance_ip <- varImp(model1, scale=FALSE)
# summarize importance
print(importance_ip)
# plot importance
plot(importance_ip)

#Should iphonesentiment be numeric or something else?
#I think it should be converted to factor because it helps 
str(df_ip)

#=================================================================
#converting sentiment column to factor 
#=================================================================
df_ip$iphonesentiment<- as.factor(df_ip$iphonesentiment) 
iphoneCOR$iphonesentiment<- as.factor(iphoneCOR$iphonesentiment)
iphoneNZV$iphonesentiment<- as.factor(iphoneNZV$iphonesentiment)
iphoneRFE$iphonesentiment<- as.factor(iphoneRFE$iphonesentiment)
str(df_ip)
#=================================================================
#"Out of the Box" Model Development
#=================================================================

#df_ip - original 
set.seed(1234)
#C5.0
#Random Forest
#SVM (from the e1071 package) 
#kknn (from the kknn package)
df_intraining <- createDataPartition(df_ip$iphonesentiment, p = .70, list = FALSE)
df_training <- df_ip[df_intraining,]
df_testing <- df_ip[-df_intraining,]
View(df_testing)
View(df_training)

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                            search = "grid")

#=================================================================
#Random Forest
#=================================================================
set.seed(1234)
rfGrid_df <- expand.grid(mtry=c(1,2,3,4,5))
caret.rf <- train(iphonesentiment ~ ., 
                  data = df_training,
                  method = "rf",
                  tuneGrid = rfGrid_df,
                  trControl = train.control)

caret.rf
plot(caret.rf)
rfImp <- varImp(caret.rf, scale = FALSE)
rfImp

#=================================================================
C5
#=================================================================
set.seed(1234)
fitControl_c5 <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3)

grid_c5 <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )

x <- df_training[,1:58]
y <- df_training$iphonesentiment
caret.c5 <- train(x=x,y=y,tuneGrid=grid_c5,trControl=fitControl_c5 ,method="C5.0",verbose=FALSE)
caret.c5 
plot(caret.c5)
c5Imp <- varImp(caret.c5, scale = FALSE)
c5Imp

#=================================================================
#SVM (from the e1071 package) 
#=================================================================
set.seed(1234)
#model_svm <- svm(formula = iphonesentiment ~ .,
 #                data = df_training,
   #              type = 'C-classification',
   #              kernel = 'linear')



trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)

svm_Linear <- train(iphonesentiment ~., data = df_training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

summary(svm_Linear)
svm_Linear
#=================================================================
#kknn (from the kknn package)
#=================================================================
set.seed(1234)
#model_kknn <- train.kknn(iphonesentiment ~ ., data = df_training, kmax = 9)
ctrl_kknn <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

kknn_fit <- train(iphonesentiment ~., data = df_training, method = "kknn",
                 trControl=ctrl_kknn,
                 preProcess = c("center", "scale"),
                 tuneLength = 10) 
View(df_training)
kknn_fit
plot(kknn_fit)

#=================================================================
#Prediction 
#=================================================================
#RF
preds_rf <- predict(caret.rf, df_testing)
#C5
preds_c5 <- predict(caret.c5, df_testing)
#SVM
preds_svm <- predict(svm_Linear, df_testing)
#KKNN
preds_kknn <- predict(kknn_fit, df_testing)
#=================================================================
#Confusion Matrix
#=================================================================
confusionMatrix(preds_rf, df_testing$iphonesentiment)
confusionMatrix(preds_c5, df_testing$iphonesentiment)
confusionMatrix(preds_svm, df_testing$iphonesentiment)
confusionMatrix(preds_kknn, df_testing$iphonesentiment)

postResample(pred = preds_rf, obs = df_testing$iphonesentiment)
postResample(pred = preds_c5, obs = df_testing$iphonesentiment)
postResample(pred = preds_svm, obs = df_testing$iphonesentiment)
postResample(pred = preds_kknn, obs = df_testing$iphonesentiment)

df_model <- data.frame(Model="rf",Accuracy=0.7501285,Kappa=0.5011703)
df_model1 <- data.frame(Model="C5",Accuracy=0.7748072,Kappa=0.5665646)
df_model2 <- data.frame(Model="SVM",Accuracy=0.7177378,Kappa=0.4339577)
df_model3<- data.frame(Model="KKNN",Accuracy=0.3694087,Kappa=0.1883560)
df_sample <-rbind(df_model,df_model1,df_model2,df_model3)
View(df_sample)


ggplot(df_sample, aes(x=Kappa, y=Model)) + 
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set1")   +
  coord_flip()

ggplot(df_sample, aes(x=Accuracy, y=Model)) + 
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set1")   +
  coord_flip()

ggplot(data = df_sample, aes( x = Model , y = Accuracy, fill = Kappa ) ) +    # print bar chart
  geom_bar( stat = 'identity', position = 'dodge' )+
  labs(list(x = "x", y = "count",fill = "group"))

resample_results <- resamples(list(RF = caret.rf, C5.0=caret.c5, svm=svm_Linear,kknn=kknn_fit))
summary(resample_results)
#=================================================================
#c5 has the most accuracy and kappa. 
#=================================================================
#iphoneCOR 
#=================================================================
set.seed(1234)
#C5.0
#Random Forest
#SVM (from the e1071 package) 
#kknn (from the kknn package)
iphoneCOR_intraining <- createDataPartition(iphoneCOR$iphonesentiment, p = .70, list = FALSE)
iphoneCOR_training <- iphoneCOR[iphoneCOR_intraining,]
iphoneCOR_testing <- iphoneCOR[-iphoneCOR_intraining,]
View(iphoneCOR_testing)
View(iphoneCOR_training)


set.seed(1234)
fitControl_c5_COR <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3)

grid_c5_COR <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )
u=iphoneCOR_training[,1:27]
caret.c5.COR <- train(u,y=iphoneCOR_training$iphonesentiment,tuneGrid=grid_c5_COR,trControl=fitControl_c5_COR ,method="C5.0",verbose=FALSE)
caret.c5.COR 
plot(caret.c5.COR )
c5Imp.COR <- varImp(caret.c5.COR, scale = FALSE)
c5Imp.COR
preds_c5_COR <- predict(caret.c5.COR, iphoneCOR_testing)
postResample(pred = preds_c5_COR , obs = iphoneCOR_testing$iphonesentiment)
#=================================================================
#iphoneNZV
#=================================================================
set.seed(1234)
#C5.0
#Random Forest
#SVM (from the e1071 package) 
#kknn (from the kknn package)
iphoneNZV_intraining <- createDataPartition(iphoneNZV$iphonesentiment, p = .70, list = FALSE)
iphoneNZV_training <- iphoneNZV[iphoneCOR_intraining,]
iphoneNZV_testing <- iphoneNZV[-iphoneCOR_intraining,]
View(iphoneNZV_testing)
View(iphoneNZV_training)
set.seed(1234)
fitControl_c5_NZV <- trainControl(method = "repeatedcv",
                                  number = 10,
                                  repeats = 3)

grid_c5_NZV <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )
v=iphoneNZV_training[,1:11]
caret.c5.NZV <- train(v,y=iphoneNZV_training$iphonesentiment,tuneGrid=grid_c5_NZV,trControl=fitControl_c5_NZV ,method="C5.0",verbose=FALSE)
caret.c5.NZV
plot(caret.c5.NZV )
c5Imp.NZV <- varImp(caret.c5.NZV, scale = FALSE)
c5Imp.NZV
preds_c5_NZV <- predict(caret.c5.NZV, iphoneNZV_testing)
postResample(pred = preds_c5_NZV , obs = iphoneNZV_testing$iphonesentiment)

#=================================================================
#RFE
#=================================================================

set.seed(1234)
#C5.0
#Random Forest
#SVM (from the e1071 package) 
#kknn (from the kknn package)
iphoneRFE_intraining <- createDataPartition(iphoneRFE$iphonesentiment, p = .70, list = FALSE)
iphoneRFE_training <- iphoneRFE[iphoneRFE_intraining,]
iphoneRFE_testing <- iphoneRFE[-iphoneRFE_intraining,]
View(iphoneRFE_testing)
View(iphoneRFE_training)
set.seed(1234)
fitControl_c5_RFE <- trainControl(method = "repeatedcv",
                                  number = 10,
                                  repeats = 3)

grid_c5_RFE <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )
w=iphoneRFE_training[,1:56]
caret.c5.RFE <- train(w,y=iphoneRFE_training$iphonesentiment,tuneGrid=grid_c5_RFE,trControl=fitControl_c5_RFE ,method="C5.0",verbose=FALSE)
caret.c5.RFE
plot(caret.c5.RFE )
c5Imp.RFE <- varImp(caret.c5.RFE, scale = FALSE)
c5Imp.RFE
preds_c5_RFE <- predict(caret.c5.RFE, iphoneRFE_testing)
postResample(pred = preds_c5_RFE , obs = iphoneRFE_testing$iphonesentiment)

# end result 
postResample(pred = preds_c5_NZV , obs = iphoneNZV_testing$iphonesentiment)
postResample(pred = preds_c5_COR , obs = iphoneCOR_testing$iphonesentiment)
postResample(pred = preds_c5_RFE , obs = iphoneRFE_testing$iphonesentiment)
postResample(pred = preds_c5, obs = df_testing$iphonesentiment)


df_ft <- data.frame(Model="C5_NZV",Accuracy=0.7609254,Kappa=0.5342354)
df_ft1 <- data.frame(Model="C5_COR",Accuracy=0.7609254,Kappa=0.5342354)
df_ft2 <- data.frame(Model="C5_RFE",Accuracy=0.7755784,Kappa=0.5661037)
df_ft3<- data.frame(Model="C5_ORG",Accuracy=0.7748072,Kappa=0.5665646)
df_ft4<- data.frame(Model="C5_RC",Accuracy=0.8544987,Kappa=0.6438301)
df_ft5<- data.frame(Model="C5_RC_RFE",Accuracy=0.8542416,Kappa=0.6430659)
data_sample  <-rbind(df_ft,df_ft1,df_ft2,df_ft3)
View(data_final)

data_final <- rbind(df_ft4,df_ft5)
ggplot(data = data_sample, aes( x = Model , y = Accuracy, fill = Kappa ) ) +    # print bar chart
  geom_bar( stat = 'identity', position = 'dodge' )+
  labs(list(x = "x", y = "count",fill = "group"))

# create a new dataset that will be used for recoding sentiment
iphoneRC <- df_ip
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(iphoneRC)
str(iphoneRC)
# make iphonesentiment a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)
set.seed(1234)
#C5.0
#Random Forest
#SVM (from the e1071 package) 
#kknn (from the kknn package)
iphoneRC_intraining <- createDataPartition(iphoneRC$iphonesentiment, p = .70, list = FALSE)
iphoneRC_training <- iphoneRC[iphoneRC_intraining,]
iphoneRC_testing <- iphoneRC[-iphoneRC_intraining,]
View(iphoneRC_testing)
View(iphoneRC_training)
set.seed(1234)
fitControl_c5_RC <- trainControl(method = "repeatedcv",
                                  number = 10,
                                  repeats = 3)

grid_c5_RC <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )
m=iphoneRC_training[,1:58]
caret.c5.RC <- train(m,y=iphoneRC_training$iphonesentiment,tuneGrid=grid_c5_RC,trControl=fitControl_c5_RC ,method="C5.0",verbose=FALSE)
caret.c5.RC
plot(caret.c5.RC)
c5Imp.RC <- varImp(caret.c5.RC, scale = FALSE)
c5Imp.RC
preds_c5_RC <- predict(caret.c5.RC, iphoneRC_testing)
postResample(pred = preds_c5_RC , obs = iphoneRC_testing$iphonesentiment)


# create a new dataset that will be used for recoding sentiment
iphoneRC_RFE <- iphoneRFE
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC_RFE$iphonesentiment <- recode(iphoneRC_RFE$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(iphoneRC_RFE)
str(iphoneRC_RFE)
# make iphonesentiment a factor
iphoneRC_RFE$iphonesentiment <- as.factor(iphoneRC_RFE$iphonesentiment)
set.seed(1234)
iphoneRC_RFE_intraining <- createDataPartition(iphoneRC_RFE$iphonesentiment, p = .70, list = FALSE)
iphoneRC_RFE_training <- iphoneRC_RFE[iphoneRC_RFE_intraining,]
iphoneRC_RFE_testing <- iphoneRC_RFE[-iphoneRC_RFE_intraining,]
View(iphoneRC_RFE_testing)
View(iphoneRC_RFE_training)
set.seed(1234)
fitControl_c5_RC_RFE <- trainControl(method = "repeatedcv",
                                 number = 10,
                                 repeats = 3)

grid_c5_RC_RFE <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )
n=iphoneRC_RFE_training[,1:56]
caret.c5.RC_RFE <- train(n,y=iphoneRC_RFE_training$iphonesentiment,tuneGrid=grid_c5_RC_RFE,trControl=fitControl_c5_RC_RFE ,method="C5.0",verbose=FALSE)
caret.c5.RC_RFE
plot(caret.c5.RC_RFE)
c5Imp.RC.RFE <- varImp(caret.c5.RC_RFE, scale = FALSE)
c5Imp.RC.RFE
preds_c5_RC.RFE <- predict(caret.c5.RC_RFE, iphoneRC_RFE_testing)
postResample(pred = preds_c5_RC.RFE , obs = iphoneRC_RFE_testing$iphonesentiment)

#Principal Component Analysis 

# data = training and testing from iphoneDF (no feature selection) 
data <- df_ip
data_intraining <- createDataPartition(data$iphonesentiment, p = .70, list = FALSE)
data_training <- data[data_intraining,]
data_testing <- data[-data_intraining,]
View(data_training)
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(data_training[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, data_training[,-59])
View(data_training[,-59])
# add the dependent to training
train.pca$iphonesentiment <- data_training$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, data_testing[,-59])
test.pca
# add the dependent to training
test.pca$iphonesentiment <- data_testing$iphonesentiment
test.pca$iphonesentiment <- as.factor(test.pca$iphonesentiment)
# inspect results
str(train.pca)
str(test.pca)

#=================================================================
#load data galaxy small matrix
#=================================================================
b <- file.choose()
df_gal <-read.csv(b)
corrData.gal <- cor(df_gal)
corrData.gal
corrplot(corrData.gal)
View(df_gal)
plot_ly(df_gal,x=~df_gal$galaxysentiment, type='histogram')
sum(duplicated(df_gal))
galaxyRC_gal <- df_gal
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
galaxyRC_gal$galaxysentiment <- recode(galaxyRC_gal$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(galaxyRC_gal)
str(galaxyRC_gal)
# make galaxy sentiment a factor
galaxyRC_gal$galaxysentiment <- as.factor(galaxyRC_gal$galaxysentiment)
set.seed(1234)

galaxyRC_gal_intraining <- createDataPartition(galaxyRC_gal$galaxysentiment, p = .70, list = FALSE)
galaxyRC_gal_training <- galaxyRC_gal[galaxyRC_gal_intraining,]
galaxyRC_gal_testing <- galaxyRC_gal[-galaxyRC_gal_intraining ,]
View(galaxyRC_gal_testing)
View(galaxyRC_gal_training)
set.seed(1234)
fitControl_c5_RC_gal <- trainControl(method = "repeatedcv",
                                     number = 10,
                                     repeats = 3)

grid_c5_RC_gal <- expand.grid( .winnow = FALSE, .trials=c(1,2,3,4,5), .model="tree" )
o=galaxyRC_gal_training[,1:58]
caret.c5.RC_gal <- train(o,y=galaxyRC_gal_training$galaxysentiment,tuneGrid=grid_c5_RC_gal,trControl=fitControl_c5_RC_gal ,method="C5.0",verbose=FALSE)
caret.c5.RC_gal
preds_c5_RC.gal <- predict(caret.c5.RC_gal, galaxyRC_gal_testing)
postResample(pred = preds_c5_RC.gal , obs = galaxyRC_gal_testing$galaxysentiment)

caret.rf.gal <- train(galaxysentiment ~ ., 
                  data = galaxyRC_gal_training,
                  method = "rf",
                  tuneGrid = rfGrid_df,
                  trControl = train.control)

caret.rf.gal
plot(caret.rf.gal)

preds_rf_RC.gal <- predict(caret.rf.gal, galaxyRC_gal_testing)
postResample(pred = preds_rf_RC.gal , obs = galaxyRC_gal_testing$galaxysentiment)

#=================================================================
#load data large matrix - iphone
#=================================================================

c <- file.choose()
df_large <-read.csv(c)

# make iphonesentiment a factor
df_large$iphonesentiment <- as.factor(df_large$iphonesentiment)

#predict 
preds_final_iphonesentiment <- predict(caret.c5.RC_RFE, df_large)
df_large$iphonesentiment<-preds_final_iphonesentiment 


View(df_large)
plot_ly(df_large, x= ~df_large$iphonesentiment, type='histogram')

confusionMatrix(preds_final_iphonesentiment,df_large$iphonesentiment)
summary(preds_final_iphonesentiment)

# create a data frame for plotting.
# you can add more sentiment levels if needed
# Replace sentiment values 
pieData <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive","positive"), 
                      values = c(10405, 963, 1326, 7318 ))
library(ggplot2)
update.packages("ggplot2")
# create pie chart
plot_ly(pieData, labels = ~COM, values = ~ values, type = "pie",
              textposition = 'inside',
              textinfo = 'label+percent',
              insidetextfont = list(color = '#FFFFFF'),
              hoverinfo = 'text',
              text = ~paste( values),
              marker = list(colors = colors,
                            line = list(color = '#FFFFFF', width = 1)),
              showlegend = F) %>%
  layout(title = 'iPhone Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

#=================================================================
#load data large matrix - galaxy 
#=================================================================

d <- file.choose()
df_large_gal <-read.csv(d)
# make galaxysentiment a factor
df_large_gal$galaxysentiment <- as.factor(df_large_gal$galaxysentiment)

#predict 
preds_final_galaxysentiment <- predict(caret.c5.RC_gal, df_large_gal)
df_large_gal$galaxysentiment<-preds_final_galaxysentiment 

View(df_large_gal)
plot_ly(df_large_gal, x= ~df_large_gal$galaxysentiment, type='histogram')

confusionMatrix(preds_final_galaxysentiment,df_large_gal$galaxysentiment)
summary(preds_final_galaxysentiment)

pieData_gal <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive","positive"), 
                      values = c(10331, 996, 1365, 7320))
par(mfrow=c(1,1))
# create pie chart
plot_ly(pieData_gal, labels = ~COM, values = ~ values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'Galaxy Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

#Side by side 


df <- data.frame(COM = c("ip", "ip","gal", "gal"), values= c(11368,8644,11327,8685))
View(df)
dataiphone <- filter(df, COM == "ip")
datagalaxy <- filter(df, COM == "gal")
View(dataiphone)
View(datagalaxy)


df <- data.frame( COM = c("PostiveIP", "PositiveGal","NegativeIP", "NegativeGal"), values= c(11368,11327,8644,8685))
plot_ly(df, labels = ~COM, values = ~ values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'Iphone/Galaxy Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)
