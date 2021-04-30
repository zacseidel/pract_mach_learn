---
title: 'Practical Machine Learning: Final Project'
output:
  html_document: 
    keep_md: yes
  pdf_document: default
---

# Executive Summary

This analysis provides an overview of the prediction results of the weightlifting dataset.  The goal was to predict the 20 observations in the test set.

That was done by:

* Preprocessing the data to remove NA's or irrelevant information (like user identification)
* Splitting the training set into a training and validation set and setting model parameters
* fitting a random forest model and using it to predict the results
 
 We find that:
 
* Accuracy from Cross Validation was >99.6%
* Accuracy in our test set was 100%
* We have a model with very strong predictive results

# Model

This data comes from the Weight Lifting Dataset within the [Human Activity Recognition Project]( http://groupware.les.inf.puc-rio.br/har)

Training data is [available here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv), and the testing data is [provided here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

First, we download the data

```r
library(tidyverse)
library(readr)
library(caret)
training <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
dim(testing);dim(training)
```

```
## [1]  20 160
```

```
## [1] 19622   160
```

The goal of the project is to predict the "classe" variable, which is the last column in the dataset.

```r
as.tibble(head(training))
```

```
## Warning: `as.tibble()` is deprecated as of tibble 2.0.0.
## Please use `as_tibble()` instead.
## The signature and semantics have changed, see `?as_tibble`.
## This warning is displayed once every 8 hours.
## Call `lifecycle::last_warnings()` to see where this warning was generated.
```

```
## # A tibble: 6 x 160
##      X1 user_name raw_timestamp_p… raw_timestamp_p… cvtd_timestamp new_window
##   <dbl> <chr>                <dbl>            <dbl> <chr>          <chr>     
## 1     1 carlitos        1323084231           788290 05/12/2011 11… no        
## 2     2 carlitos        1323084231           808298 05/12/2011 11… no        
## 3     3 carlitos        1323084231           820366 05/12/2011 11… no        
## 4     4 carlitos        1323084232           120339 05/12/2011 11… no        
## 5     5 carlitos        1323084232           196328 05/12/2011 11… no        
## 6     6 carlitos        1323084232           304277 05/12/2011 11… no        
## # … with 154 more variables: num_window <dbl>, roll_belt <dbl>,
## #   pitch_belt <dbl>, yaw_belt <dbl>, total_accel_belt <dbl>,
## #   kurtosis_roll_belt <chr>, kurtosis_picth_belt <chr>,
## #   kurtosis_yaw_belt <chr>, skewness_roll_belt <chr>,
## #   skewness_roll_belt.1 <chr>, skewness_yaw_belt <chr>, max_roll_belt <dbl>,
## #   max_picth_belt <dbl>, max_yaw_belt <chr>, min_roll_belt <dbl>,
## #   min_pitch_belt <dbl>, min_yaw_belt <chr>, amplitude_roll_belt <dbl>,
## #   amplitude_pitch_belt <dbl>, amplitude_yaw_belt <chr>,
## #   var_total_accel_belt <dbl>, avg_roll_belt <dbl>, stddev_roll_belt <dbl>,
## #   var_roll_belt <dbl>, avg_pitch_belt <dbl>, stddev_pitch_belt <dbl>,
## #   var_pitch_belt <dbl>, avg_yaw_belt <dbl>, stddev_yaw_belt <dbl>,
## #   var_yaw_belt <dbl>, gyros_belt_x <dbl>, gyros_belt_y <dbl>,
## #   gyros_belt_z <dbl>, accel_belt_x <dbl>, accel_belt_y <dbl>,
## #   accel_belt_z <dbl>, magnet_belt_x <dbl>, magnet_belt_y <dbl>,
## #   magnet_belt_z <dbl>, roll_arm <dbl>, pitch_arm <dbl>, yaw_arm <dbl>,
## #   total_accel_arm <dbl>, var_accel_arm <dbl>, avg_roll_arm <dbl>,
## #   stddev_roll_arm <dbl>, var_roll_arm <dbl>, avg_pitch_arm <dbl>,
## #   stddev_pitch_arm <dbl>, var_pitch_arm <dbl>, avg_yaw_arm <dbl>,
## #   stddev_yaw_arm <dbl>, var_yaw_arm <dbl>, gyros_arm_x <dbl>,
## #   gyros_arm_y <dbl>, gyros_arm_z <dbl>, accel_arm_x <dbl>, accel_arm_y <dbl>,
## #   accel_arm_z <dbl>, magnet_arm_x <dbl>, magnet_arm_y <dbl>,
## #   magnet_arm_z <dbl>, kurtosis_roll_arm <dbl>, kurtosis_picth_arm <chr>,
## #   kurtosis_yaw_arm <chr>, skewness_roll_arm <dbl>, skewness_pitch_arm <chr>,
## #   skewness_yaw_arm <chr>, max_roll_arm <dbl>, max_picth_arm <dbl>,
## #   max_yaw_arm <dbl>, min_roll_arm <dbl>, min_pitch_arm <dbl>,
## #   min_yaw_arm <dbl>, amplitude_roll_arm <dbl>, amplitude_pitch_arm <dbl>,
## #   amplitude_yaw_arm <dbl>, roll_dumbbell <dbl>, pitch_dumbbell <dbl>,
## #   yaw_dumbbell <dbl>, kurtosis_roll_dumbbell <dbl>,
## #   kurtosis_picth_dumbbell <dbl>, kurtosis_yaw_dumbbell <chr>,
## #   skewness_roll_dumbbell <dbl>, skewness_pitch_dumbbell <dbl>,
## #   skewness_yaw_dumbbell <chr>, max_roll_dumbbell <dbl>,
## #   max_picth_dumbbell <dbl>, max_yaw_dumbbell <dbl>, min_roll_dumbbell <dbl>,
## #   min_pitch_dumbbell <dbl>, min_yaw_dumbbell <dbl>,
## #   amplitude_roll_dumbbell <dbl>, amplitude_pitch_dumbbell <dbl>,
## #   amplitude_yaw_dumbbell <dbl>, total_accel_dumbbell <dbl>,
## #   var_accel_dumbbell <dbl>, avg_roll_dumbbell <dbl>,
## #   stddev_roll_dumbbell <dbl>, var_roll_dumbbell <dbl>, …
```

```r
as.tibble(head(testing))
```

```
## # A tibble: 6 x 160
##      X1 user_name raw_timestamp_p… raw_timestamp_p… cvtd_timestamp new_window
##   <dbl> <chr>                <dbl>            <dbl> <chr>          <chr>     
## 1     1 pedro           1323095002           868349 05/12/2011 14… no        
## 2     2 jeremy          1322673067           778725 30/11/2011 17… no        
## 3     3 jeremy          1322673075           342967 30/11/2011 17… no        
## 4     4 adelmo          1322832789           560311 02/12/2011 13… no        
## 5     5 eurico          1322489635           814776 28/11/2011 14… no        
## 6     6 jeremy          1322673149           510661 30/11/2011 17… no        
## # … with 154 more variables: num_window <dbl>, roll_belt <dbl>,
## #   pitch_belt <dbl>, yaw_belt <dbl>, total_accel_belt <dbl>,
## #   kurtosis_roll_belt <lgl>, kurtosis_picth_belt <lgl>,
## #   kurtosis_yaw_belt <lgl>, skewness_roll_belt <lgl>,
## #   skewness_roll_belt.1 <lgl>, skewness_yaw_belt <lgl>, max_roll_belt <lgl>,
## #   max_picth_belt <lgl>, max_yaw_belt <lgl>, min_roll_belt <lgl>,
## #   min_pitch_belt <lgl>, min_yaw_belt <lgl>, amplitude_roll_belt <lgl>,
## #   amplitude_pitch_belt <lgl>, amplitude_yaw_belt <lgl>,
## #   var_total_accel_belt <lgl>, avg_roll_belt <lgl>, stddev_roll_belt <lgl>,
## #   var_roll_belt <lgl>, avg_pitch_belt <lgl>, stddev_pitch_belt <lgl>,
## #   var_pitch_belt <lgl>, avg_yaw_belt <lgl>, stddev_yaw_belt <lgl>,
## #   var_yaw_belt <lgl>, gyros_belt_x <dbl>, gyros_belt_y <dbl>,
## #   gyros_belt_z <dbl>, accel_belt_x <dbl>, accel_belt_y <dbl>,
## #   accel_belt_z <dbl>, magnet_belt_x <dbl>, magnet_belt_y <dbl>,
## #   magnet_belt_z <dbl>, roll_arm <dbl>, pitch_arm <dbl>, yaw_arm <dbl>,
## #   total_accel_arm <dbl>, var_accel_arm <lgl>, avg_roll_arm <lgl>,
## #   stddev_roll_arm <lgl>, var_roll_arm <lgl>, avg_pitch_arm <lgl>,
## #   stddev_pitch_arm <lgl>, var_pitch_arm <lgl>, avg_yaw_arm <lgl>,
## #   stddev_yaw_arm <lgl>, var_yaw_arm <lgl>, gyros_arm_x <dbl>,
## #   gyros_arm_y <dbl>, gyros_arm_z <dbl>, accel_arm_x <dbl>, accel_arm_y <dbl>,
## #   accel_arm_z <dbl>, magnet_arm_x <dbl>, magnet_arm_y <dbl>,
## #   magnet_arm_z <dbl>, kurtosis_roll_arm <lgl>, kurtosis_picth_arm <lgl>,
## #   kurtosis_yaw_arm <lgl>, skewness_roll_arm <lgl>, skewness_pitch_arm <lgl>,
## #   skewness_yaw_arm <lgl>, max_roll_arm <lgl>, max_picth_arm <lgl>,
## #   max_yaw_arm <lgl>, min_roll_arm <lgl>, min_pitch_arm <lgl>,
## #   min_yaw_arm <lgl>, amplitude_roll_arm <lgl>, amplitude_pitch_arm <lgl>,
## #   amplitude_yaw_arm <lgl>, roll_dumbbell <dbl>, pitch_dumbbell <dbl>,
## #   yaw_dumbbell <dbl>, kurtosis_roll_dumbbell <lgl>,
## #   kurtosis_picth_dumbbell <lgl>, kurtosis_yaw_dumbbell <lgl>,
## #   skewness_roll_dumbbell <lgl>, skewness_pitch_dumbbell <lgl>,
## #   skewness_yaw_dumbbell <lgl>, max_roll_dumbbell <lgl>,
## #   max_picth_dumbbell <lgl>, max_yaw_dumbbell <lgl>, min_roll_dumbbell <lgl>,
## #   min_pitch_dumbbell <lgl>, min_yaw_dumbbell <lgl>,
## #   amplitude_roll_dumbbell <lgl>, amplitude_pitch_dumbbell <lgl>,
## #   amplitude_yaw_dumbbell <lgl>, total_accel_dumbbell <dbl>,
## #   var_accel_dumbbell <lgl>, avg_roll_dumbbell <lgl>,
## #   stddev_roll_dumbbell <lgl>, var_roll_dumbbell <lgl>, …
```

Many of the columns contain 100% NA's in the test set.  We exclude the columns that have no NA's, as well as the 5x columns as the beginning of the dataset that identify users.

```r
cols <- colnames(training[,colMeans(is.na(testing))==0])
cols <- cols[-c(1:5)]
cols
```

```
##  [1] "new_window"           "num_window"           "roll_belt"           
##  [4] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [7] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
## [10] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [13] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [16] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [19] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [22] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [25] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [28] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [31] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [34] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [37] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [40] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [43] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [46] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [49] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [52] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
## [55] "classe"
```

```r
trainCols <- training[,cols]
dim(trainCols)
```

```
## [1] 19622    55
```
Next, we split this training set into a training and test set, so that we can have an estimate of out of sample error.


```r
set.seed(13234)

inTrain <- createDataPartition(y = trainCols$classe, p = 0.8, list = FALSE)

trainColsPre <- trainCols[inTrain,]
```

```
## Warning: The `i` argument of ``[`()` can't be a matrix as of tibble 3.0.0.
## Convert to a vector.
## This warning is displayed once every 8 hours.
## Call `lifecycle::last_warnings()` to see where this warning was generated.
```

```r
trainColsTest <- trainCols[-inTrain,]

dim(trainColsPre); dim(trainColsTest)
```

```
## [1] 15699    55
```

```
## [1] 3923   55
```


Next, we set the classe variable as a factor, rather than a character.

```r
trainColsPre$classe <- as.factor(trainColsPre$classe)
trainColsTest$classe <- as.factor(trainColsTest$classe)
```

We then define our training control parameters, to perform a limited number of cross validation iterations to reduce the computational time required.

```r
fitControl <- trainControl(method = "cv", number = 10)
```

We then fit our random forest model.  I chose random forest because it tends to be the most accurate model type.  I limited the number of trees to 10 to reduce the processing time for the model

```r
modfit <- train(classe ~ ., method = "rf", data = trainCols, trControl = fitControl, ntree = 10)
```


```r
modfit
```

```
## Random Forest 
## 
## 19622 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17659, 17658, 17662, 17660, 17659, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9902155  0.9876218
##   28    0.9970439  0.9962608
##   54    0.9943434  0.9928452
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 28.
```
Next, we check this model on the test set to get an estimate of out of sample error.

```r
predtest <- predict(modfit, newdata = trainColsTest)
pred.acc <- sum(predtest == trainColsTest$classe)/length(trainColsTest$classe)
pred.acc
```

```
## [1] 1
```

Since this has 100% accuracy on the test set, and 99.6% accurracy in the cross validation, we think this is a strong model.

Next we predict on the final test set.

First we make sure the columns match

```r
cols[55]
```

```
## [1] "classe"
```

```r
predcols <- cols[-55]
dim(testing[,predcols])
```

```
## [1] 20 54
```

Next we predict using our model.

```r
predfit <- predict(modfit, newdata = testing[,predcols])
```



```r
length(predfit)
```

```
## [1] 20
```

```r
predfit
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

This is our solution.

We can plot the cross validation accuracy to confirm what our calculations showed.

```r
plot(modfit, which = 1)
```

![](FinalProject_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

