library(lasso2)
library(ISLR)
library(leaps)
library(dplyr)
library(caret)
library(bootstrap)
library(boot)
library(ElemStatLearn)


set.seed(420)
#data("Prostate")
df_prostate = ElemStatLearn::prostate

summary(df_prostate)

training_samples_cont = df_prostate$train == 'TRUE'
train_data_cont = df_prostate[training_samples_cont, ]
train_data_cont = train_data_cont[,-10]
test_data_cont = df_prostate[!training_samples_cont, ]
test_data_cont = test_data_cont[,-10]
Y.train = df_prostate$lpsa[training_samples_cont]
Y.test = df_prostate$lpsa[!training_samples_cont]

####################### Linear Regression ###################################################
colnames(df_prostate)
linear_fit <- lm(lpsa~., data = train_data_cont)
summary(linear_fit)

fit <- regsubsets(lpsa~., data = train_data_cont, method = "exhaustive", nvmax = 8)
summary_subset <- summary(fit)
names(summary_subset)
cp = summary_subset$cp
cp #7 variable model is the best
bic = summary_subset$bic
bic #2 variable model is the best

number_vars_cp = which.min(cp)
number_vars_bic = which.min(bic)

select = summary(fit)$outmat
train.error.store <- c()
test.error.store <- c()

######### train and test error using best Cp statistic which is for 7 variables ###################

vars <- which(select[number_vars_cp,] == "*")
red.training <- train_data_cont[, c(9,vars)]
red.testing <- test_data_cont[,c(9,vars)]
red.fit <- lm(lpsa~., data = red.training)
pred.train = predict(red.fit, newdata = red.training)
pred.test = predict(red.fit, newdata = red.testing)
test.error <- (1/length(Y.test))*sum((pred.test - Y.test)^2)
train.error <- (1/length(Y.train))*sum((pred.train - Y.train)^2)
#train.error.store <- c(train.error.store, train.error)
#test.error.store <- c(test.error.store, test.error)

######### train and test error using best BIC statistic which is for 3 variables ###################

vars <- which(select[number_vars_bic,] == "*")
red.training <- train_data_cont[, c(9,vars)]
red.testing <- test_data_cont[,c(9,vars)]
red.fit <- lm(lpsa~., data = red.training)
pred.train = predict(red.fit, newdata = red.training)
pred.test = predict(red.fit, newdata = red.testing)
test.error <- (1/length(Y.test))*sum((pred.test - Y.test)^2)
train.error <- (1/length(Y.train))*sum((pred.train - Y.train)^2)
#train.error.store <- c(train.error.store, train.error)
#test.error.store <- c(test.error.store, test.error)

###### train and test error using 5 fold cross validation #########################  

select = summary(fit)$outmat
train.error.store <- c()
test.error.store <- c()

train.control <- trainControl(method = "cv", number = 5)

for (i in 1:8){
  temp <- which(select[i,] == "*")
  temp <- temp + 1
  
  red.training <- train_data_cont[, c(9,temp)]
  red.testing <- test_data_cont[,c(9,temp)]
  
  red.fit <- train(lpsa~., data = red.training, method = "lm",
                   trControl = train.control)
  
  pred.train = predict(red.fit, newdata = red.training)
  pred.test = predict(red.fit, newdata = red.testing)
  
  test.error <- (1/length(Y.test))*sum((pred.test - Y.test)^2)
  train.error <- (1/length(Y.train))*sum((pred.train - Y.train)^2)
  
  train.error.store <- c(train.error.store, train.error)
  test.error.store <- c(test.error.store, test.error)
  
}

upper = max(train.error.store, test.error.store)
lower = min(train.error.store, test.error.store)

#quartz()
plot(train.error.store, type = "o", lty = 2, col = "blue", ylim = c(lower -0.5, upper +0.05) , xlab = "#variables",
     ylab = "error", main = "Model Selection for 5 fold CV")
lines(test.error.store, type = "o", lty = 1, col = "red")
legend("topright", c("training", "test"), lty = c(2,1), col = c("blue", "red"))

###### train and test error using 10 fold cross validation #########################  

select = summary(fit)$outmat
train.error.store.10 <- c()
test.error.store.10 <- c()

train.control <- trainControl(method = "cv", number = 10)

for (i in 1:8){
  temp <- which(select[i,] == "*")
  temp <- temp + 1
  
  red.training <- train_data_cont[, c(9,temp)]
  red.testing <- test_data_cont[,c(9,temp)]
  
  red.fit <- train(lpsa~., data = red.training, method = "lm",
                   trControl = train.control)
  
  pred.train = predict(red.fit, newdata = red.training)
  pred.test = predict(red.fit, newdata = red.testing)
  
  red.fit$results
  test.error <- (1/length(Y.test))*sum((pred.test - Y.test)^2)
  train.error <- (1/length(Y.train))*sum((pred.train - Y.train)^2)
  
  train.error.store.10 <- c(train.error.store, train.error)
  test.error.store.10 <- c(test.error.store, test.error)
}


upper = max(train.error.store.10, test.error.store.10)
lower = min(train.error.store.10, test.error.store.10)

#quartz()
plot(train.error.store.10, type = "o", lty = 2, col = "blue", ylim = c(lower -0.5, upper +0.5) , xlab = "#variables",
     ylab = "Error", main = "Model Selection for 10 fold CV")
lines(test.error.store.10, type = "o", lty = 1, col = "red")
legend("topright", c("training", "test"), lty = c(2,1), col = c("blue", "red"))

######################## bootstrap ########################

# create functions that feed into "bootpred"
beta.fit <- function(X,Y){
  lsfit(X,Y)	
}

beta.predict <- function(fit, X){
  cbind(1,X)%*%fit$coef
}

sq.error <- function(Y,Yhat){
  (Y-Yhat)^2
}

# Create X and Y
X <- df_prostate[,1:8]
Y <- df_prostate[,9]

error_store <- c()
for (i in 1:8){
  # Pull out the model
  temp <- which(select[i,] == "*")
  
  res <- bootpred(X[,temp], Y, nboot = 50, theta.fit = beta.fit, theta.predict = beta.predict, err.meas = sq.error) 
  error_store <- c(error_store, res[[3]])
  
}

upper = max(error_store)
lower = min(error_store)

plot(error_store, type = "o", lty = 2, col = "blue", ylim = c(lower-0.1, upper+0.1),
     xlab = "#variables",ylab = "error", main = "Model Selection using Bootstrap")

