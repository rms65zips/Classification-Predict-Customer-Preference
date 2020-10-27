# Title: R_caret_pipeline_basic_regression_WholeYear

# Last update: 2019.09

# File/project name: 2019 R-pipeline-caret-basic-regression-WholeYear.R
# RStudio Project name: See resources for details on R projects

###############
# Project Notes
###############


# Summarize project: This is a simple R pipeline to help understand how to organize
# a project in R Studio using the the caret package. This is a basic pipeline that
# does not include feature selection (filtering/wrapper methods/embedded methods).  

# Summarize top model and/or filtered dataset

# Objects
# x <- 5 + 3 + 6  
# y <- 10
# x
# y
# x + y


# Assignment "<-" short-cut: 
#   OSX [Alt]+[-] (next to "+" sign)
#   Win [Alt]+[-] 


# Comment multiple lines
# OSX: CTRL + SHIFT + C
# WIN: CMD + SHIFT + C


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
# set working directory
setwd("D:/UT Data Analytics/Course 2 - Predicting Customer Preferences/Task2 - Classification_Predict which Brand of Products Customers Prefer/Task2_Surveys")
dir()

# set a value for seed (to be used in the set.seed function)
seed <- 123


################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
library(caret)
library(corrplot)
#library(doMC)
library(doParallel)
library(mlbench)
library(readr)


#####################
# Parallel processing
#####################

# NOTE: Be sure to use the correct package for your operating system.

#--- for OSX ---#
#install.packages("doMC")  # install in 'Load packages' section above 
#library(doMC)
#detectCores()   # detect number of cores
#registerDoMC(cores = 4)  # set number of cores; 2 in this example (don't use all available)

#--- for WIN ---#
install.packages("doParallel") # install in 'Load packages' section above
library(doParallel)  # load in the 'Load Packages' section above
detectCores()  # detect number of cores
cl <- makeCluster(2)  # select number of cores; 2 in this example
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

##############
# Import data
##############

#### --- Load raw datasets --- ####

# --- Load Train/Existing data (Dataset 1) --- #
CompleteResponses <- read.csv("CompleteResponses.csv", stringsAsFactors = FALSE)
class(CompleteResponses)  # "data.frame"
str(CompleteResponses)



# --- Load Predict/New data (Dataset 2) --- #

SurveyIncomplete <- read.csv("SurveyIncomplete.csv", stringsAsFactors = FALSE)
class(SurveyIncomplete)
str(SurveyIncomplete)



#### --- Load preprocessed datasets --- ####

#ds_name <- read.csv("dataset_name.csv", stringsAsFactors = FALSE) 


################
# Evaluate data
################

#--- Dataset 1 ---#
str(CompleteResponses)  # 35136 obs. of  8 variables 
names(CompleteResponses)
summary(CompleteResponses)
head(CompleteResponses)
tail(CompleteResponses)

# plot
hist(CompleteResponses$salary)
plot(CompleteResponses$brand, CompleteResponses$salary)
qqnorm(CompleteResponses$salary)
# check for missing values 
anyNA(CompleteResponses)
is.na(CompleteResponses)
anyNA(SurveyIncomplete)
is.na(SurveyIncomplete)

#--- Dataset 2 ---#

# If there is a dataset with unseen data to make predictions on, then preprocess here
# to make sure that it is preprossed the same as the training dataset.


#############
# Preprocess
#############

#--- Dataset 1 ---#

# change data types
CompleteResponses$brand <- as.factor(CompleteResponses$brand)
summary(CompleteResponses)
str(CompleteResponses$brand)
# rename a column
names(CompleteResponses)[names(CompleteResponses) == "elevel"] <- "education" 
summary(CompleteResponses)

#--- Dataset 2 ---#

# change data types
SurveyIncomplete$brand <- as.factor(SurveyIncomplete$brand)
summary(SurveyIncomplete)
str(SurveyIncomplete$brand)

# rename a column
names(SurveyIncomplete)[names(SurveyIncomplete) == "elevel"] <- "education" 
summary(SurveyIncomplete)



str(CompleteResponses) # 9898 obs. of  7 variables

# save preprocessed dataset
#write.csv()


################
# Feature Selection
################

## ---- Corr analysis ----- ###


################
# Sampling
################

# ---- Sampling ---- #

# Note: The set.seed function has to be executed immediately preceeding any 
# function that needs a seed value

# Note: For this task, use the 1000 sample, and not the 10%

# 1k sample
set.seed(seed)
CompleteResponses1k <- CompleteResponses[sample(1:nrow(CompleteResponses), 1000, replace=FALSE),]
head(CompleteResponses1k) # ensure randomness
nrow(CompleteResponses1k) # ensure number of obs
# create 10% sample for 7v -- 990 obs
set.seed(seed) # set random seed
CompleteResponses10p <- CompleteResponses[sample(1:nrow(CompleteResponses), round(nrow(CompleteResponses)*.1),replace=FALSE),]
nrow(CompleteResponses10p) # 990 obs
head(CompleteResponses10p) # ensure randomness


##################
# Train/test sets
##################

# create the training partition that is 75% of total obs
set.seed(seed) # set random seed
inTraining <- createDataPartition(CompleteResponses1k$brand, p=0.75, list=FALSE)
# create training/testing dataset
trainSet <- CompleteResponses1k[inTraining,]   
testSet <- CompleteResponses1k[-inTraining,]   
# verify number of obs 
nrow(trainSet)  
nrow(testSet)   
str(trainSet)


################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
summary(fitControl)

##############
# Train model
##############

set.seed(seed)
tree_mod <- train(brand~., data=trainSet, method="C5.0",importance=T, trControl=fitControl)

varImp(tree_mod)

# C5.0 variable importance
# 
# Overall
# salary    100.000
# credit     98.375
# age        95.524
# education  77.560
# car         4.077
# zipcode     0.000

plot(tree_mod)

tree_mod


## ------- RF ------- ##

# RF train/fit

set.seed(seed)
rfGrid <- expand.grid(mtry=c(1,2,3,4,5))
system.time(rfFit1 <- train(brand~., data=trainSet, method="rf", importance=T, trControl=fitControl,tuneGrid=rfGrid)) #importance is needed for varImp


rfFit1
summary(rfFit1)

plot(rfFit1)
varImp(rfFit1, scale = FALSE)

# rf variable importance
# 
# Importance
# salary       100.000
# age           72.125
# credit         2.307
# education      1.753
# car            1.722
# zipcode        0.000


#################
# Evaluate models
#################

##--- Compare models ---##

# use resamples to compare model performance
ModelFitResults1k <- resamples(list(C5.0=tree_mod, rf=rfFit1))

# output summary metrics for tuned models 
summary(ModelFitResults1k)

# Call:
#   summary.resamples(object = ModelFitResults1k)
# 
# Models: C5.0, rf 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5.0 0.7066667 0.7773684 0.8600000 0.8456140 0.9066667 0.9466667    0
# rf   0.8400000 0.8933333 0.9066667 0.9014561 0.9169737 0.9333333    0
# 
# Kappa 
#           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5.0 0.3569758 0.5384913 0.7060070 0.6757242 0.8068214 0.8861048    0
# rf   0.6790300 0.7758199 0.8044693 0.7939305 0.8216413 0.8567062    0

##########################################################################


#Top performing model.

#Random Forest 

rfFit1

#751 samples
# 6 predictor
# 2 classes: '0', '1' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 675, 676, 676, 676, 676, 676, ... 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa    
# 1     0.7428772  0.4286887
# 2     0.8668070  0.7194699
# 3     0.8921053  0.7741066
# 4     0.9001228  0.7910825
# 5     0.9014561  0.7939305
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 5.

# rf variable importance
# 
# Importance
# salary       100.000
# age           72.125
# credit         2.307
# education      1.753
# car            1.722
# zipcode        0.000


##--- Conclusion ---##
# The Random Forest model shows a higher average kappa and accuracy value.


########################
# Validate top model
########################

################# C5.0 Predictions ################

lmPred1 <- predict(tree_mod, testSet)

postResample(lmPred1,testSet$brand)
# Accuracy     Kappa 
# 0.9156627 0.8263771 

# plot predicted verses actual
plot(lmPred1,testSet$brand)

lmPred1

summary(lmPred1)
# 0   1 
# 109 140 

################# RF Predictions ##################

# make predictions
rfPred1 <- predict(rfFit1, testSet)

postResample(rfPred1, testSet$brand) # performace measurment
 
# Accuracy     Kappa 
# 0.9598394 0.9155646 

# plot predicted verses actual
plot(rfPred1,testSet$brand)
plot(testSet$brand,rfPred1)
rfPred1

# [1] 0 1 1 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 0 0 1 0 0 0 0 0
# [38] 0 0 0 1 0 0 1 1 1 1 1 0 1 1 0 1 1 1 0 1 1 1 1 1 1 0 1 0 0 0 1 0 1 1 0 1 1
# [75] 1 1 0 1 1 1 0 1 0 1 0 1 1 0 1 0 0 1 1 1 0 1 0 0 0 1 1 0 0 0 1 0 1 0 0 0 1
# [112] 1 0 1 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 0 0 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1
# [149] 0 0 1 1 0 1 0 1 1 0 1 0 0 1 0 1 1 1 0 0 0 1 0 1 1 0 0 1 1 1 1 0 0 0 1 1 0
# [186] 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 1 1 0 1 0 1 0 0 1 1 1 1 1
# [223] 0 0 0 1 0 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0
# Levels: 0 1

########################
# Predict with top model
########################

### C5.0 Predictions ###

# lmPred2 <- predict(tree_mod, SurveyIncomplete)
# lmPred2
# summary(lmPred2)
# 0    1 
# 1997 3003 


### RF Predictions ###

rfPred2 <- predict(rfFit1, SurveyIncomplete)
rfPred2
plot(rfPred2)
summary(rfPred2)
# 0    1 
# 1879 3121



########################
# Save validated model
########################


##--- Save top performing model ---##
write.csv(rfPred2, "rfPred2.csv")

# save model 
saveRDS(rfFit1, file = "rfFit1.rds")  #saveRDS saves the representation of the object so you can load into a differently named object within R.

# load and name model to make predictions with new data
RFfit1 <- readRDS(file = "rfFit1.rds") #readRDS deserializes the object and converts it back to a character type.




