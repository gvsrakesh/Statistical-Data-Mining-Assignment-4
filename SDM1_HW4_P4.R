library(lasso2)
library(ISLR)
library(leaps)
library(dplyr)
library(caret)
library(bootstrap)
library(boot)  
library(kernlab)
library(MASS)
library(mlbench)
library(randomForest)

set.seed(420)
data(spam)
df_spam = spam
training_samples_cont = df_spam$type %>% createDataPartition(p = 0.75, list = FALSE)
train_data_cont = df_spam[training_samples_cont, ]
test_data_cont = df_spam[-training_samples_cont, ]

m = c(2,5,7,8,10,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57)
error_df = as.data.frame(m)
oob_list = c()
test_error_list = c()
for (i in 1:length(m))
{
  rf_fit = randomForest(type~., data = train_data_cont, n.tree = 500,
                        na.action = na.roughfix, mtry = m[i],do.trace = 100)
  oob = mean(rf_fit$err.rate[,1])
  test_pred <- predict(rf_fit, newdata = test_data_cont, type = "response")
  CM = confusionMatrix(test_data_cont$type, test_pred)
  error_df$test_error[i] = 1 - CM$overall[1]
  error_df$oob[i] = oob
}

varImpPlot(rf_fit)

lower = min(error_df$oob,error_df$test_error)
upper = max(error_df$oob,error_df$test_error)
plot(error_df$m,error_df$oob, type = "o", lty = 2, col = "blue", ylim = c(lower -0.05, upper +0.05) ,
     xlab = "mtry values", ylab = "Error", main = "OOB and Test error vs mtry values")
lines(error_df$m,error_df$test_error, type = "o", lty = 1, col = "red")
legend("topright", c("OOB", "Test Error"), lty = c(2,1), col = c("blue", "red"))

