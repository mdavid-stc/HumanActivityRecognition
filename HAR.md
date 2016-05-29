# Human Activity Recognition
This is the course project for Practical Machine Learning.

## Loading and preprocessing the data
The data is assumed to be in a CSV file in the same directory as this document. If it is not, download it and read it into a data frame.

```r
knitr::opts_chunk$set(fig.path='figure/')
options(scipen = 1, digits = 2)

# Check for the presence of the data.
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainfn <- "pml-training.csv"

# Make sure the data is available in the current directory.
if (!file.exists(trainfn)) {
    # Download the data file
    download.file(trainUrl, trainfn, "curl")
}

training <- read.csv(trainfn, na.strings=c("NA",""))

# Check for the presence of the data.
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testfn <- "pml-testing.csv"

# Make sure the data is available in the current directory.
if (!file.exists(testfn)) {
    # Download the data file
    download.file(testUrl, testfn, "curl")
}

testing <- read.csv(testfn, na.strings=c("NA",""))
```

Comparing the training set names to the testing set names (the only thing I looked at in the test set before predicting), we see that the outcome variable in the training set ("classe") is replaced by a "problem_id" variable in the testing set to give us a way to refer to specific rows in our prediction.
The training set is 19622 observations, which is quite a lot when we're predicting only 20 test cases, so we have some room to drop troublesome observations and to subset for testing.

Looking at the summaries of the training variables, some are obviously poor, so remove these (and also remove from testing so that the two sets are still comparable).


```r
# The row index, user's name, and timestamps cannot have predictive value.
badvars <- c(1:5)
# There is not enough variation in new_window to help, so we'll exclude that also.
badvars <- c(badvars, 6)
# Many variables are ~98% NA or blank, so remove them.
for (i in 1:length(training)) {
    if (sum(is.na(training[i])) >= (nrow(training)*0.95)) {
        badvars <- c(badvars, i)
    }
}

training <- training[, -badvars]
testing <- testing[, -badvars]
# Down to 54 variables.
```

A side effect of trimming down the data has gotten rid of all NA values in the data set. All of the remaining 54 variables are numeric or integer types (except the 5-level outcome, classe), and they are all complete (that is, we did not sacrifice any rows in the above data reduction).


## Building a model

Now split the training set into a further training and validation sets since we're not allowed to figure this out using the real testing data (and we have plenty of observations).


```r
library(caret)
library(ggplot2)
set.seed(777)
```


```r
# First, check that there are enough samples in each outcome class.
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
# A fairly even split, so partitioning won't have any problems.
trainIndex <- createDataPartition(training$classe, p = 0.75, list = FALSE)
subtrain <- training[trainIndex, ]
validate <- training[-trainIndex, ]
```

Since we're predicting a factor variable, a Decision Tree would be an appropriate tool, but due to the complexity of this problem, a Random Forest will almost certainly be better. Use 5-fold cross validation to reduce overfitting.


```r
# This was taking a long time, so I added allowParallel.
fit1 <- train(classe ~ ., data=subtrain, method="rf", trControl=trainControl(method="cv", number=5), allowParallel=TRUE)
# Do I need prox=TRUE?
```


```r
fit1
```

```
## Random Forest 
## 
## 14718 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11776, 11776, 11774, 11773, 11773 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa
##    2    0.99      0.99 
##   27    1.00      1.00 
##   53    1.00      0.99 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
fit1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.26%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4183    1    0    0    1     0.00048
## B    7 2837    3    1    0     0.00386
## C    0    6 2560    1    0     0.00273
## D    0    0   12 2399    1     0.00539
## E    0    1    0    4 2701     0.00185
```

Theoretically, that is an excellent model, with >99% accuracy. Let's take a look at how it performs on the validation data.

```r
subpred <- predict(fit1, newdata=validate)
confusionMatrix(subpred, validate$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  946    1    0    0
##          C    0    1  854    5    0
##          D    0    0    0  799    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.999    0.994    1.000
## Specificity             0.999    1.000    0.999    1.000    1.000
## Pos Pred Value          0.999    0.999    0.993    1.000    1.000
## Neg Pred Value          1.000    0.999    1.000    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.193    0.175    0.163    0.184
## Balanced Accuracy       1.000    0.998    0.999    0.997    1.000
```

Again, fantastic numbers (only 9 wrong out of 4904, 99.8% accurate), so I don't feel a need to adjust the model (which we could still do at this point since we still haven't looked at the real test data), but let's look into the elements of the model a bit further.


## Variable Importance

Now that we have our model, let's find out which variables are most important.

```r
varImp(fit1, scale=FALSE)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 53)
## 
##                      Overall
## num_window              2183
## roll_belt               1351
## pitch_forearm            874
## yaw_belt                 648
## magnet_dumbbell_y        627
## magnet_dumbbell_z        618
## pitch_belt               593
## roll_forearm             507
## accel_dumbbell_y         308
## magnet_dumbbell_x        246
## accel_belt_z             236
## roll_dumbbell            234
## accel_forearm_x          226
## total_accel_dumbbell     197
## magnet_forearm_z         172
## accel_dumbbell_z         170
## magnet_belt_y            170
## magnet_belt_z            159
## magnet_belt_x            138
## yaw_dumbbell             118
```
Unexpectedly, the num_window variable is hugely significant (more than 50% more important than the next)!


## Plots of Important Predictors

Let's look at plots of the three most important variables to see how they relate to the outcome.

```r
qplot(x=num_window, y=roll_belt, data = subtrain, col=classe)
```

![](figure/unnamed-chunk-9-1.png)<!-- -->

```r
qplot(x=num_window, y=pitch_forearm, data = subtrain, col=classe)
```

![](figure/unnamed-chunk-9-2.png)<!-- -->

```r
qplot(x=roll_belt, y=pitch_forearm, data = subtrain, col=classe)
```

![](figure/unnamed-chunk-9-3.png)<!-- -->

The second plot especially shows a very strong correlation of num_window with classe *regardless of the y variable*.


## Single-Variable Model

Since one general goal is always to make our models as parsimonious as possible, let's try a model using just that predictor.


```r
# Doesn't make sense to cross validate with one variable, so that option is off here.
fit2 <- train(classe ~ num_window, data=subtrain, method="rf", allowParallel=TRUE)
fit2
```

```
## Random Forest 
## 
## 14718 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## Resampling results:
## 
##   Accuracy  Kappa
##   1         1    
## 
## Tuning parameter 'mtry' was held constant at a value of 2
## 
```

```r
fit2$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 1
## 
##         OOB estimate of  error rate: 0.01%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4185    0    0    0    0     0.00000
## B    0 2847    1    0    0     0.00035
## C    0    0 2567    0    0     0.00000
## D    0    0    0 2412    0     0.00000
## E    0    0    0    1 2705     0.00037
```

```r
subpred2 <- predict(fit2, newdata=validate)
confusionMatrix(subpred2, validate$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  948    0    0    0
##          C    0    0  855    0    0
##          D    0    0    0  804    0
##          E    0    1    0    0  901
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.999    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    0.999
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.999    1.000    1.000    1.000
```

It turns out that num_window on its own is an almost perfect predictor, misclassifying only two observations in our (sub)training set and only one observation in the (sub)test set.

Given the data partitioning and cross validation used, the expected out of sample error is very small (less than 0.01%).


## Dropping the Window Number Predictor

The num_window variable is not one of the features talked about in the description of the data collection setup and doesn't seem to be a measured variable, so it's likely that it's not meant to be used for predicting. So let's build one more model without it.

```r
fit3 <- train(classe ~ . - num_window, data=subtrain, method="rf", trControl=trainControl(method="cv", number=5), allowParallel=TRUE)
fit3
```

```
## Random Forest 
## 
## 14718 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11772, 11775, 11775, 11775, 11775 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa
##    2    0.99      0.99 
##   27    0.99      0.99 
##   52    0.98      0.98 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
fit3$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4182    1    1    0    1     0.00072
## B   13 2830    5    0    0     0.00632
## C    0   21 2542    4    0     0.00974
## D    0    0   43 2367    2     0.01866
## E    0    0    0    4 2702     0.00148
```

```r
subpred3 <- predict(fit3, newdata=validate)
confusionMatrix(subpred3, validate$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1392    7    0    0    0
##          B    0  941    9    0    0
##          C    3    1  846   18    0
##          D    0    0    0  786    1
##          E    0    0    0    0  900
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.989, 0.994)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.99          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.992    0.989    0.978    0.999
## Specificity             0.998    0.998    0.995    1.000    1.000
## Pos Pred Value          0.995    0.991    0.975    0.999    1.000
## Neg Pred Value          0.999    0.998    0.998    0.996    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.160    0.184
## Detection Prevalence    0.285    0.194    0.177    0.160    0.184
## Balanced Accuracy       0.998    0.995    0.992    0.989    0.999
```

This model's accuracy was not as good, but still >99% so still more than good enough.


## Final Prediction

Now we're ready to look at the test set.
So the final prediction for the project is:

```r
prediction <- predict(fit3, newdata=testing)
data.frame(testing$problem_id, prediction)
```

```
##    testing.problem_id prediction
## 1                   1          B
## 2                   2          A
## 3                   3          B
## 4                   4          A
## 5                   5          A
## 6                   6          E
## 7                   7          D
## 8                   8          B
## 9                   9          A
## 10                 10          A
## 11                 11          B
## 12                 12          C
## 13                 13          B
## 14                 14          A
## 15                 15          E
## 16                 16          E
## 17                 17          A
## 18                 18          B
## 19                 19          B
## 20                 20          B
```

As expected, the prediction is perfect (20 for 20)!

