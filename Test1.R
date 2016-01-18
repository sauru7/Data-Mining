
setwd("D:/Study/Mini2/Data Mining/Project/Team H")


suggestions <- import.csv("suggestions.csv")

dim(suggestions)
#Dimensions are 16429 * 10

nrow(suggestions[complete.cases(suggestions),])
#All rows have values for all the attributes

#Prediction for Recommended 0 or 1
#So we use the Binomial Logistic Regression

#The variable Recommended can be a factor
library(plyr)
suggestions <- transform(suggestions, Recommended = as.factor(mapvalues(Recommended,c(0,1),c("No","Yes"))))

variables.with.missing <- sapply(suggestions, function(x) sum(is.na(x)))
#No Missing Values

#To plot the missing values vs the observed values
install.packages("Amelia")
library(Amelia)
missmap(suggestions)



#We can remove the variables Suggestion_ID and Author_ID as they dont rely make
#any sense in predicting for a recommended suggestion
suggs <- subset(suggestions, select = -which(colnames(suggestions) %in% c("Suggestion_Id","Author_Id")))
colnames(suggs)[6] <- "Author_Age"
colnames(suggestions)[8] <- "Author_Age"
#We will perform k fold cross validation
install.packages("caret")
library(caret)

#Give the indices to the createFolds method and will return the number of folds 
#as a list with random indices getting picked
indices <- 1:nrow(suggs)
folds <- createFolds(indices, k = 5, list = TRUE)
#folds
#length(folds)


#Get the testing data 
testing_all <- lapply(folds, function(x) suggs[x,])
training_all <- lapply(folds, function(x) suggs[-x,])
length(testing_all)
class(testing_all)
length(training_all)

get_pred_knn <- function(train,test,k)
{
  
  train_in <- train[,1:ncol(train) - 1]
  train_out <- train[,ncol(train)]
  test_in <- test[,1:ncol(test) - 1]
  
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



getBestGLM <- function(train,test,column)
{
  formula <- as.formula(paste("Recommended ~ ", colnames(suggs)[column], sep = " "))
  print(formula)
  cat("Sdsd", class(train))
  glm.model <- glm(formula, data = train)
  glm.predict <- predict(test$Recommended, glm.model$fitted_values)
  
  diff <- glm.predict - test$Recommended
  diff <- diff ^ 2
  MSE <- mean(diff)
  
  return (MSE)
}

y <- which(colnames(suggs) == "Recommended")
best.model.step1 <- sapply(seq_len(ncol(suggs)), function(column)
  {
    column_name <- names(suggs)[column]
    
     if(which(colnames(suggs) == column_name) != y)
     {

        MSE <- mapply(train = training_all, test = testing_all, getBestGLM(train,test,column))
        
        min_MSE_for_all_models <- min(unlist(MSE))
        
        return (min_MSE_for_all_models)
     }
     else
     {
       return (NULL)
     }
      
  })

#A General logistic Regression model with all variables included
log.suggs <- glm(Recommended ~ ., data = suggs, family = binomial(logit))
summary(log.suggs)

#Shows that all variables except for Author_TotalPosts is helpful in determing the Recommended


#VotesUp does not affect, see the plot
log.suggs <- glm(Recommended ~ Votes_Up, data = suggs, family = binomial(logit))
#summary(log.suggs)
summary(log.suggs)
plot(Recommended ~ Votes_Up , data = suggs)
lines(suggs$Votes_Up, log.suggs$fitted.values, type = "l")

log.suggs <- glm(Recommended ~ Votes_Down, data = suggs, family = binomial(logit))
summary(log.suggs)
plot(Recommended ~ Votes_Down , data = suggs)
lines(suggs$Votes_Down, log.suggs$fitted.values, type = "l")

log.suggs <- glm(Recommended ~ Responses, data = suggestions, family = binomial(logit))
#summary(log.suggs)
plot(Recommended ~ Responses , data = suggs)
lines(suggs$Responses, log.suggs$fitted, type = "l")

log.suggs <- glm(Recommended ~ Author_PostsPerDay, data = suggs, family = binomial(logit))
#summary(log.suggs)
plot(Recommended ~ Author_PostsPerDay , data = suggs)
lines(suggs$Author_PostsPerDay, log.suggs$fitted, type = "l")

log.suggs <- glm(Recommended ~ Author_PostsPerDay + Votes_Up, data = suggs, family = binomial(logit))
#summary(log.suggs)
plot(Recommended ~ Author_PostsPerDay + Votes_Up , data = suggs)



sapply(df1, function(column){
  
  column_name <- colnames(column)
  print(column_name)
  if(which(colnames(df1) %in% column_name) != 1)
  {
    print("sds")
  }
  else
  {
    print("qqqq")
  }
  
})





get_pred_knn <- function(train,test,k)
{
  
  train_in <- train[,1:ncol(train) - 1]
  train_out <- train[,ncol(train)]
  test_in <- test[,1:ncol(test) - 1]
  
  pred_values <- knn(train = train_in, test = test_in, cl = train_out, k = k,prob = TRUE , use.all = FALSE)
  predicted <- attr(pred_values,"prob")
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
  df_knn <- data.frame(pred_values = predicted_high, orig_values = test[,ncol(test)])
  
  
  return (df_knn)
}

get_pred_glm <- function(train,test)
{
  output_column <- names(train)[ncol(train)]
  formula <- as.formula(paste(output_column, "~", ".", sep = " "))
  pred_glm <- glm(formula, data = train, family = binomial())
  #if I give without type = response the probabilities are on the logit scale (log-odds)
  #if I give type as response the probabilities are on the normal scale
  predicted <- predict(pred_glm,test, type = "response")
  #levels(test[,ncol(test)]) <- levels(test[,ncol(test)])[c(2,1)]
  #test[,ncol(test)] <- as.numeric(test[,ncol(test)])
  df_logreg <- data.frame(pred_values = predicted, orig_values = test[,ncol(test)])
  
  return (df_logreg)
}