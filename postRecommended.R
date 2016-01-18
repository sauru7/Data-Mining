setwd("D:/Study/Mini2/Data Mining/Project/Team H")

library(plyr)
library(leaps)
library(MASS)
library(class)
library(e1071)
library(caret)
library(mlbench)
library(randomForest)
library(rpart)
library(ROCR)


install.packages("penalizedSVM")
install.packages("rknn")
library(penalizedSVM)
library(rknn)

posts <- import.csv("posts.csv")


#posts <- transform(posts, Recommended = as.factor(mapvalues(Recommended,c(0,1),c("No","Yes"))))
posts$Recommended <- factor(posts$Recommended)

posts_original <- posts[,c(1:9)]
posts_original <- subset(posts_original, select = -which(colnames(posts) %in% c("SuggID","AuthID")))
posts_subset <- subset(posts, select = -which(colnames(posts) %in% c("SuggID","AuthID")))


#Finding the best features via exhaustive search
#Dont use this as this does feature selection on the entire data set and then 
#does model building on the training data, which is essentially wrong
#always keep feature selection and model building process based on training data
#Here what happened was for feature selection we included testing data as well
#exhaustive_best_features <- regsubsets(x = posts[,-1], y = posts[,1], nbest = 1, nvmax = 14)
#summary(exhaustive_best_features)

#Making output column as the last column
n <- ncol(posts)
posts <- posts[,c(2:n,1)]

n <- ncol(posts_subset)
posts_subset <- posts_subset[,c(2:n,1)]

n <- ncol(posts_original)
posts_original <- posts_original[,c(2:n,1)]

#Selecting the best features for random forest
#control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
#feature_results <- rfe(posts[,c(1:(n-1))],posts[,n], sizes = c(1:(n-1)),rfeControl = control)

#Selecting the best features for naive bayes
#control <- rfeControl(functions = nbFuncs, method = "cv", number = 10)
#feature_results_nb <- rfe(posts[,c(1:(n-1))],posts[,n], sizes = c(1:(n-1)),rfeControl = control)

#Selecting the best features for SVM
#control <- rfeControl(functions = caretFuncs, method = "cv", number = 10)
#feature_results_svm <- rfe(posts[,c(1:(n-1))],posts[,n], sizes = c(1:(n-1)),rfeControl = control, method = "svmRadial")


get_pred_default <- function(train,test)
{
  #Get all high and low values
  high <- length(which(train[,ncol(train)] == "1"))
  low <- length(which(train[,ncol(train)] == '0'))
  max <- c()
  
  #Return the probability of the max level, The level occurring more number of times
  if(high > low)
  {
    max = high/(high+low)
  }
  else
  {
    max = low/(high+low)
    max = 1 - max
  }
    
  predicted = rep(max, nrow(test))
  df_default <- data.frame(pred_values = predicted, orig_values = test[,ncol(test)])
  
  return (df_default)
}

get_pred_knn <- function(train,test,k)
{
  
  #train_in <- train[,1:ncol(train) - 1]
  train_out <- train[,ncol(train)]
  #test_in <- test[,1:ncol(test) - 1]
  train_in <- train[,c(4,5,8,11,12,13,14)]
  test_in <- test[,c(4,5,8,11,12,13,14)]
  #Using the knn from the class library to predict the values for test data
  pred_values <- knn(train = train_in, test = test_in, cl = train_out, k = k,prob = TRUE, use.all = FALSE)
  #Get the probabilities associated with class
  
  
  
  predicted <- attr(pred_values,"prob")
  
  #knn returns probability with the associated class
  #So we determine the probability for high
  #and if the class returned low, then use 1 - probability
  predicted_high <- sapply(1:length(pred_values), function(x){
    if(pred_values[x] == 1)
    {
      return (predicted[x])
    }
    else
    {
      return (1- predicted[x])
    }
  })
  
  #Return a dataframe of predicted values and original values
  df_knn <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  
  return (df_knn)
}

get_pred_logreg <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", ".", sep = " "))
  #Run the logistic regression function
  pred_glm <- glm(formula, data = train, family = binomial())
  
  #if I give without type = response the probabilities are on the logit scale (log-odds)
  #if I give type as response the probabilities are on the normal scale
  predicted <- predict(pred_glm,test, type = "response")
  
  #Reorder the levels as predict for glm model takes first level as the base level
  #and predicts the probability for the next level
  #Hence we reorder and get level 0 first and level 1 after that
  #levels(test[,ncol(test)]) <- levels(test[,ncol(test)])[c(2,1)]
  
  #Return a dataframe of predicted values and original values
  df_logreg <- data.frame(pred_values = predicted, orig_values = test[,ncol(test)])
  
  return (df_logreg)
}
get_pred_logregStepAIC <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", ".", sep = " "))
  #Run the logistic regression function
  pred_glm <- glm(formula, data = train, family = binomial())
  
  stepaic <- stepAIC(pred_glm, direction = "both")
  
  features_var <- attr(terms(stepaic), "term.labels")
  print(stepaic$anova)
  x <-  c()
  l <- length(features_var)
  for(i in 1:length(features_var))
  {
    if(i == (l))
    {
      x <- paste(x, eval(features_var[i]), "", sep = " ")
    }
    else
    {
      x <- paste(x,eval(features_var[i]), "+", sep = " ")
    }
  }
  formula <- as.formula(paste(output_column,"~",x,sep = " "))
  pred_glm <- glm(formula, data = train,family = binomial())
  #if I give without type = response the probabilities are on the logit scale (log-odds)
  #if I give type as response the probabilities are on the normal scale
  predicted <- predict(pred_glm,test, type = "response")
  
  #Reorder the levels as predict for glm model takes first level as the base level
  #and predicts the probability for the next level
  #Hence we reorder and get level 0 first and level 1 after that
  #levels(test[,ncol(test)]) <- levels(test[,ncol(test)])[c(2,1)]
  
  #Return a dataframe of predicted values and original values
  df_logreg <- data.frame(pred_values = predicted, orig_values = test[,ncol(test)])
  
  return (df_logreg)
}


#Support Vector Machine Classifier
get_pred_svm <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", "SuggID + Responses + Views + VoteUp + VoteDown + AuthScore  + Up.Views + Down.Views", sep = " "))
  #Run the SVM Model
  pred_svm <- svm(formula, data = train, probability = TRUE)
  
  #Get the predicted values for the testing data
  predicted <- predict( pred_svm, test, probability = TRUE)
  
  #Take the probability of class '1' that is level = High
  index_col_1 <- which(colnames(attr(predicted,"probabilities")) %in% "1")
  predicted_high <- attr(predicted,"probabilities")[,index_col_1]
  
  #Return a dataframe of predicted values and original values
  df_svm <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_svm)
}


get_pred_svm_greedyForward <- function(train,test)
{
  n <- ncol(train)
  output_column <- names(train)[n]
  getmetrics <- c()
  indexi <- c()
  feature <- c()
  best_fea <- c()
  final_fea <- c()
  
  for(j in 1:(n-1))
  {
    
    
    for(i in 1:(n-1))
    {
      x <- c()
      
      feature <- names(train)[i]
      
      l <- length(final_fea)
      if(l == 0)
      {
        
        x <- paste(final_fea,"", eval(feature), sep = " ")
      }
      else
      {
        x <- paste(final_fea,"+",eval(feature), sep = " ")
      }
      
      #cat(x)
      formula <- as.formula(paste(output_column, "~", x, sep = " "))
      pred_svm <- svm(formula, data = train, probability = TRUE)
      predicted <- predict( pred_svm, test, probability = TRUE)
      index_col_1 <- which(colnames(attr(predicted,"probabilities")) %in% "1")
      predicted_high <- attr(predicted,"probabilities")[,index_col_1]
      df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
      getmetrics[i] <- get_metrics(df_nb,0.5)$true.pos.rate
      best_fea[i] <- x
    }
    #cat(getmetrics)
    print(getmetrics)
    indexi <- which.max(getmetrics)
    #print(indexi)
    cat("Best", best_fea[indexi])
    final_fea <- best_fea[indexi]
    print(getmetrics[indexi])
    
    if(getmetrics[indexi] == 1)
    {
      stop()
    }
  }
  #Return a dataframe of predicted values and original values
  #df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_nb)
}



#Support Vector Machine Classifier with feature selection
get_pred_svm_fs <- function(train,test)
{
  n <- ncol(train)
  output_column <- names(train)[ncol(train)]
  
  formula <- as.formula(paste(output_column, "~", "AuthTenure", sep = " "))
  #Run the SVM Model
  pred_svm <- svm.fs(x= train[,1:n], y = train[,n])
  #Get the predicted values for the testing data
  predicted <- predict( pred_svm, test, probability = TRUE)
  
  #Take the probability of class '1' that is level = High
  index_col_1 <- which(colnames(attr(predicted,"probabilities")) %in% "1")
  predicted_high <- attr(predicted,"probabilities")[,index_col_1]
  
  #Return a dataframe of predicted values and original values
  df_svm <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_svm)
}

#Naive Bayes Classifier 
get_pred_nb <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", "Responses + VoteUp + AuthTenure + AuthScore + Up.Down + Up.Views + Down.Views", sep = " "))
  
  #Get he classification model
  pred_nb <- naiveBayes(formula, data = train, type = "raw")
  
  #Predict the values based on our new model on the testing data
  predicted <- predict( pred_nb, test, type = "raw")
  
  #Get the probability for class '1' that is level - High
  index_col_1 <- which(colnames(predicted) %in% "1")
  predicted_high <- predicted[,index_col_1]
  
  #Return a dataframe of predicted values and original values
  df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_nb)
}

#Naive Bayes Classifier with greedy forward selection

get_pred_nb_greedyForward <- function(train,test)
{
  n <- ncol(train)
  output_column <- names(train)[n]
  getmetrics <- c()
  indexi <- c()
  feature <- c()
  best_fea <- c()
  final_fea <- c()
  
  for(j in 1:(n-1))
  {
    
    
    for(i in 1:(n-1))
    {
      x <- c()
      
      feature <- names(train)[i]
      
      l <- length(final_fea)
        if(l == 0)
        {
          
          x <- paste(final_fea,"", eval(feature), sep = " ")
        }
        else
        {
          x <- paste(final_fea,"+",eval(feature), sep = " ")
        }
      
      #cat(x)
      formula <- as.formula(paste(output_column, "~", x, sep = " "))
      pred_nb <- naiveBayes(formula, data = train, type = "raw")
      predicted <- predict( pred_nb, test, type = "raw")
      index_col_1 <- which(colnames(predicted) %in% "1")
      predicted_high <- predicted[,index_col_1]
      df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
      getmetrics[i] <- get_metrics(df_nb,0.5)$true.pos.rate
      best_fea[i] <- x
    }
    #cat(getmetrics)
    print(getmetrics)
    indexi <- which.max(getmetrics)
    #print(indexi)
    cat("Best", best_fea[indexi])
    final_fea <- best_fea[indexi]
    print(getmetrics[indexi])
    
    if(getmetrics[indexi] == 1)
    {
      exit()
    }
  }
  #Return a dataframe of predicted values and original values
  #df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_nb)
}



#Decision Tree Classifier 
get_pred_dectree <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", ".", sep = " "))
  
  #Get the classification model
  pred_dt <- rpart(formula, data = train, method = "class")
  
  
  #Predict the values based on our new model on the testing data
  predicted <- predict( pred_dt, test, type = "prob")
 
  
  #Get the probability for class '1' that is level - High
  index_col_1 <- which(colnames(predicted) %in% "1")
  predicted_high <- predicted[,index_col_1]
  
  #Return a dataframe of predicted values and original values
  df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_nb)
}

#Random Forest Classifier 
get_pred_rf <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", ".", sep = " "))
  
  #Get he classification model
  pred_rf <- randomForest(formula, data = train)
  
  #Predict the values based on our new model on the testing data
  predicted <- predict( pred_rf, test, type = "prob")
  
  
  #Get the probability for class '1' that is level - High
  index_col_1 <- which(colnames(predicted) %in% "1")
  predicted_high <- predicted[,index_col_1]
  
  
  #Return a dataframe of predicted values and original values
  df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_nb)
}


#Random Forest Classifier with selected top 5 features
get_pred_rf_topfeatures <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", "VoteUp + Views + Up.Down + Down.Views + Responses", sep = " "))
  
  #Get he classification model
  pred_rf <- randomForest(formula, data = train)
  
  #Predict the values based on our new model on the testing data
  predicted <- predict( pred_rf, test, type = "prob")
  
  
  #Get the probability for class '1' that is level - High
  index_col_1 <- which(colnames(predicted) %in% "1")
  predicted_high <- predicted[,index_col_1]
  
  
  #Return a dataframe of predicted values and original values
  df_nb <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  return (df_nb)
}

do_cv_class <- function(dataframe, num_folds, model_name)
{
  #Randomize the data to remove any ordering
  dataframe <- dataframe[sample(nrow(dataframe)),]
  
  #Number of rows of the dataframe
  n <- nrow(dataframe)
  
  #Create the folds and accordingly create testing and training data
  all_folds <- lapply(c(1:num_folds), function(x) {
    
    if(num_folds == 1)
    {
      testing <- dataframe
      training <- dataframe
      return (list(test = testing, train = training))
    }
    start <- round((n/num_folds) * (x-1)) + 1
    if(x == num_folds)
    {
      end <- n
    }
    else
    {
      end <- round((n/num_folds) * x)  
    }
    indices <- start:end
    testing <- dataframe[indices,]
    training <- dataframe[-indices,]
    return (list(test = testing, train = training))
  })
  
  #The list of all testing data (length of list will be number of folds)
  testing_all <- lapply(1:num_folds, function(x){
    return(all_folds[[x]]['test'][[1]])
  })
  #The list of all training data (length of list will be number of folds)
  training_all <- lapply(1:num_folds, function(x){
    return(all_folds[[x]]['train'][[1]])
  })
  
  
  #Check if the model is knn and extract number of neighbors accordingly
  if(grepl("[0-9]*nn", model_name))
  {
    
    k <- gsub("[^[:digit:]]*" , "" , model_name)
    k <- as.integer(k)
    functionname <- paste("get","pred","knn",sep = "_")
    
    #Call the the function knn for the number of folds
    pred_res <- mapply(function(x,y,z) do.call(functionname, list(x,y,z)), x = training_all,y = testing_all, z = k, SIMPLIFY = FALSE)
  }
  else
  {
    #Create the function name based on the model_name passed
    functionname <- paste("get","pred",model_name,sep = "_")
    
    #Call the models logreg, svm or naiveBayes based on the column name
    pred_res <- mapply(function(x,y) do.call(functionname, list(x,y)), x = training_all,y = testing_all, SIMPLIFY = FALSE)
    
  }
  
  #Give the names of the list, which has the a list of dataframes
  names(pred_res) <- c(1:length(pred_res))
  
  #Get all the dataframes from the list and get the it combined
  df2 <- do.call("rbind",pred_res)
  
  return (df2)
  
}

my_pred_nb <- do_cv_class(posts,10,"nb")
my_pred_svm <- do_cv_class(posts,10,"svm")
my_pred_glm <- do_cv_class(posts,10,"logreg")
my_pred_knn <- do_cv_class(posts,10,"5nn")
my_pred_default <- do_cv_class(posts,10,"default")
my_pred_dectree <- do_cv_class(posts,10,"dectree")
my_pred_rf <- do_cv_class(posts,10,"rf")
my_pred_logregStepAIC <- do_cv_class(posts,10,"logregStepAIC")
#my_pred_rf_topfeatures <- do_cv_class(posts_subset,10,"rf_topfeatures")
#my_pred_nb_greedyForward <- do_cv_class(posts_subset,10,"nb_greedyForward")
#my_pred_svm_greedyForward <- do_cv_class(posts_subset,10,"svm_greedyForward")
get_metrics <- function(pred_df, cutoff)
{
  
  #Based on the cutoff threshold assign the appropriate class
  #for our predicted levels
  pred_df[,1] <- ifelse((pred_df[,1] > cutoff), 1,0)
  pred_df[,1] <- as.factor(pred_df[,1])
  
  #Build the confusion matrix   
  table1 <- table(pred_df[,1],pred_df[,2])
  
  
  #For default predictor
  if(nrow(table1) == 1)
  {
    tpr = fpr <- 1
    acc <- table1[1,1]/sum(table1)
    precision <- table1[1,1]/sum(table1)
    recall <- 1
    return( data.frame(true.pos.rate = tpr, false.pos.rate = fpr, accurracy = acc, precision = precision, recall = recall))
    
  }
  
  #Get names and index of all of them
  indexHighR <- which(rownames(table1) == "1")
  indexLowR <- which(rownames(table1) == "0")
  indexHighC <- which(colnames(table1) == "1")
  indexLowC <- which(colnames(table1) == "0")
  
  #Calculate true positive rate
  tpr <- table1[indexHighR,indexHighC]/sum(table1[,indexHighC])
  #Calculate False positive rate
  fpr <- table1[indexHighR,indexLowC]/sum(table1[,indexLowC])
  #Calcualte the accuracy
  acc <- (table1[indexHighR,indexHighC] + table1[indexLowR,indexLowC])/sum(table1)
  #Calcualte the precision
  precision <- table1[indexHighR,indexHighC]/sum(table1[indexHighR,])
  #Calcualte the recall
  recall <- table1[indexHighR,indexHighC]/sum(table1[,indexHighC])
  
  #Return a dataframe with all the values
  return( data.frame(true.pos.rate = tpr, false.pos.rate = fpr, accurracy = acc, precision = precision, recall = recall))
  
}


#ROC curves for all models
par(new=FALSE)

#nb
rocr_nb <- prediction(my_pred_nb[,1],my_pred_nb[,2])
nb_perf <- performance(rocr_nb, measure = "tpr",x.measure = "fpr")
#plot(nb_perf, col = "2")

semilog <- nb_perf@x.values[[1]]
options(scipen = 22)
plot(semilog, nb_perf@y.values[[1]], type = "l", log="x", col="1", xlab="", ylab= "", xaxt='n', yaxt = 'n')
par(new=TRUE)



#logreg
rocr_logreg <- prediction(my_pred_logregStepAIC[,1],my_pred_logregStepAIC[,2])
logreg_perf <- performance(rocr_logreg, measure = "tpr",x.measure = "fpr")
#plot(logreg_perf, col = "3")

semilog <- logreg_perf@x.values[[1]]
options(scipen = 22)
plot(semilog, logreg_perf@y.values[[1]], type = "l", log="x", col="2", xlab="", ylab= "")

par(new=TRUE)

#svm
rocr_svm <- prediction(my_pred_svm[,1],my_pred_svm[,2])
svm_perf <- performance(rocr_svm, measure = "tpr",x.measure = "fpr")
#plot(svm_perf, col = "3")

semilog <- svm_perf@x.values[[1]]
options(scipen = 22)
plot(semilog, svm_perf@y.values[[1]], type = "l", log="x", col="3", xlab="", ylab= "", xaxt='n', yaxt = 'n')
par(new=TRUE)

#knn
rocr_knn <- prediction(my_pred_knn[,1],my_pred_knn[,2])
knn_perf <- performance(rocr_knn, measure = "tpr",x.measure = "fpr")
#plot(knn_perf, col = "3")

semilog <- knn_perf@x.values[[1]]
options(scipen = 22)
plot(semilog, knn_perf@y.values[[1]], type = "l", log="x", col="4", xlab="", ylab= "", xaxt='n', yaxt = 'n')
par(new=TRUE)

#dectree
rocr_dectree <- prediction(my_pred_dectree[,1],my_pred_dectree[,2])
dectree_perf <- performance(rocr_dectree, measure = "tpr",x.measure = "fpr")
#plot(dectree_perf, col = "3")

semilog <- dectree_perf@x.values[[1]]
options(scipen = 22)
plot(semilog, dectree_perf@y.values[[1]], type = "l", log="x", col="5", xlab="", ylab= "", xaxt='n', yaxt = 'n')
par(new=TRUE)

#rf
rocr_rf <- prediction(my_pred_rf[,1],my_pred_rf[,2])
rf_perf <- performance(rocr_rf, measure = "tpr",x.measure = "fpr")
#plot(rf_perf, col = "3")

semilog <- rf_perf@x.values[[1]]
options(scipen = 22)
plot(semilog, rf_perf@y.values[[1]], type = "l", log="x", col="6", xlab="", ylab= "", xaxt='n', yaxt = 'n')
par(new=TRUE)

#default
rocr_default <- prediction(my_pred_default[,1],my_pred_default[,2])
default_perf <- performance(rocr_default, measure = "tpr",x.measure = "fpr")
#plot(default_perf, col = "3")

semilog <- default_perf@x.values[[1]]
options(scipen = 22)
plot(semilog, default_perf@y.values[[1]], type = "l", log="x", col="7", xlab="", ylab= "", xaxt='n', yaxt = 'n')
par(new=TRUE)

title(main="ROC for all Classifiers", 
      xlab="False Positive Rate(Natural Log scale)", ylab="True Positive Rate")
par(new=TRUE)

#axis(1, at = c(0.0001,0.0010,0.0100,0.1000,1.0000), las=1)
par(new=TRUE)

legend("topleft", c("Naive Bayes", "Logistic Regression", "SVM", "KNN", "Decision Tree","Random Forest","Default"), cex=1.0, bty="n",
       col=c(1,2,3,4,5,6,7), lty = c(1,1,1,1,1,1,1), lwd = c(1,1,1,1,1,1,1))


#Get the metrics for all the models
mterics_df_nb <- get_metrics(my_pred_nb, 0.5)
cat("\n------------NAIVE BAYES-----------------\n")
print(mterics_df_nb)

mterics_df_svm <- get_metrics(my_pred_svm, 0.5)
cat("\n------------Support Vector Machines-----------------\n")
print(mterics_df_svm)

mterics_df_glm <- get_metrics(my_pred_logregStepAIC, 0.5)
cat("\n------------LOGISTIC REGRESSION-----------------\n")
print(mterics_df_glm)

mterics_df_knn <- get_metrics(my_pred_knn, 0.5)
cat("\n------------KNN -----------------\n")
print(mterics_df_knn)

mterics_df_default <- get_metrics(my_pred_default, 0.5)
cat("\n------------DEFAULT MODEL-----------------\n")
print(mterics_df_default)

mterics_df_dectree <- get_metrics(my_pred_dectree, 0.5)
cat("\n------------DECISION TREE MODEL-----------------\n")
print(mterics_df_dectree)

mterics_df_rf <- get_metrics(my_pred_rf_topfeatures, 0.5)
cat("\n------------RANDOM FOREST MODEL-----------------\n")
print(mterics_df_rf)
