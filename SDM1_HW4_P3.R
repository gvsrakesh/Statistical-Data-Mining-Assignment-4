library(lasso2)
library(ISLR)
library(leaps)
library(dplyr)
library(caret)
library(bootstrap)
library(boot)  
library(MASS)
library(mlbench)
library(randomForest)
library(gbm)
library(class)

data("Ionosphere")
df_BC = Ionosphere
summary(df_BC)
set.seed(420)

getwd()

training_samples_cont = df_BC$Class %>% createDataPartition(p = 0.63, list = FALSE)
train_data_cont = df_BC[training_samples_cont, ]
train_data_cont = train_data_cont[,-2]
test_data_cont = df_BC[-training_samples_cont, ]
test_data_cont = test_data_cont[,-2]
y_train = as.factor(as.numeric(train_data_cont$Class)-1)
y_test = as.factor(as.numeric(test_data_cont$Class)-1)

y_train_rf = train_data_cont$Class
y_test_rf = test_data_cont$Class
############## logistic regression #########################

logit_fit = glm(Class~., data = train_data_cont, family = "binomial")
summary(logit_fit)
logit_probs_train = predict(logit_fit, newdata = train_data_cont, type = "response")
logit_probs = predict(logit_fit, newdata = test_data_cont, type = "response")
logit_pred = as.factor(ifelse(logit_probs > 0.5, 1, 0))
logit_pred_train = as.factor(ifelse(logit_probs_train > 0.5, 1, 0))

confusionMatrix(y_test, logit_pred)
confusionMatrix(y_train, logit_pred_train)

############## KNN classification ##############################

knn_error = data.frame(kvalue= numeric(0), train_er = numeric(0), test_er = numeric(0))
train_data_cont_knn = train_data_cont
test_data_cont_knn = test_data_cont

train_data_cont_knn$Class = ifelse(train_data_cont$Class == "good",1,0)
test_data_cont_knn$Class = ifelse(test_data_cont$Class == "good",1,0)

train_accuracy = list()
test_accuracy = list()

for (k in c(1,3,5,7,9,11,13,15,17,19,21,23,25))
{
  knn_pred_train <- knn(train = train_data_cont_knn,test = train_data_cont_knn,cl = train_data_cont_knn$Class, k=k)
  knn_pred_test = knn(train = train_data_cont_knn, test = test_data_cont_knn, cl=train_data_cont_knn$Class, k = k)
  summary(knn_pred_train)
  cat("for k value = ", k)
  train_table = CrossTable(x=train_data_cont_knn$Class, y = knn_pred_train, prop.chisq = FALSE)
  test_table = CrossTable(x=test_data_cont_knn$Class, y = knn_pred_test, prop.chisq = FALSE)
  test_accuracy = test_table$prop.tbl[1,1] + test_table$prop.tbl[2,2]
  train_accuracy = train_table$prop.tbl[1,1] + train_table$prop.tbl[2,2]
  knn_error = rbind(knn_error, c(k, train_accuracy, test_accuracy))
}

colnames(knn_error) = c("kvalue","train_accuracy","test_accuracy")

knn_error$accuracy_diff = knn_error$test_accuracy - knn_error$train_accuracy

############## random forest #########################
rf_fit = randomForest(Class~., data = train_data_cont, n.tree = 500, na.action = na.roughfix)

varImpPlot(rf_fit)
importance(rf_fit)

probs_train = predict(rf_fit, newdata = train_data_cont, type = "response")
probs_test = predict(rf_fit, newdata = test_data_cont, type = "response")

confusionMatrix(y_test_rf, probs_test)
confusionMatrix(y_train_rf, probs_train)

############## Bagging #########################
rf_fit_bag = randomForest(Class~., data = train_data_cont, n.tree = 500, na.action = na.roughfix, mtry = 33)

varImpPlot(rf_fit_bag)
importance(rf_fit_bag)

probs_train = predict(rf_fit_bag, newdata = train_data_cont, type = "response")
probs_test = predict(rf_fit_bag, newdata = test_data_cont, type = "response")

confusionMatrix(y_test_rf, probs_test)
confusionMatrix(y_train_rf, probs_train)

############## Boosting ##############################

boost_train = train_data_cont;
boost_train$Class = as.numeric(boost_train$Class)-1
boost_test = test_data_cont;
boost_test$Class = as.numeric(boost_test$Class)-1

d = c(1,3,5,7,9)
s = c(0.001,0.005,0.01,0.05,0.1)

accuracy_df = data.frame(d= numeric(0), s= numeric(0),train_er = numeric(0), test_er = numeric(0))

for (i in 1:length(d))
{
  for (j in 1:length(s))
  {
    boost_fit = gbm(Class~., data = boost_train, n.trees = 500, interaction.depth = d[i], shrinkage = s[j],
                    distribution = "adaboost", cv.folds = 5)
    probs_train = predict(boost_fit, newdata = boost_train, n.trees = 500, type = "response")
    probs_test = predict(boost_fit, newdata = boost_test, n.trees = 500, type = "response")
    
    gbm_pred_train = as.factor(ifelse(probs_train > 0.5, 1, 0))
    gbm_pred_test = as.factor(ifelse(probs_test > 0.5, 1, 0))
    
    cm_test = confusionMatrix(y_test, gbm_pred_test)
    cm_train = confusionMatrix(y_train, gbm_pred_train)
    train_accuracy = cm_train$overall['Accuracy']
    test_accuracy = cm_test$overall['Accuracy']
    
    accuracy_df = rbind(accuracy_df, c(d[i],s[j], train_accuracy, test_accuracy))
  }
}

colnames(accuracy_df) = c("depth","lambda","train_accuracy","test_accuracy")
