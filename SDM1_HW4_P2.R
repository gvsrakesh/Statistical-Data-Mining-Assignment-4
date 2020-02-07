library(rattle.data)
library(lasso2)
library(ISLR)
library(leaps)
library(dplyr)
library(caret)
library(bootstrap)
library(boot)
library(rpart)
library(MASS)
library("tree")
library(descr)
library(rpart.plot)
library("RColorBrewer")

set.seed(420)
df_wine = rattle.data::wine
df_wine %>% count(Type)
# 1 is Barolo, 2 is Grignolino and 3 is Barbera

training_samples_cont = df_wine$Type %>% createDataPartition(p = 0.6, list = FALSE)
train_data_cont = df_wine[training_samples_cont, ]
train_data_cont %>% count(Type)
test_data_cont = df_wine[-training_samples_cont, ]
test_data_cont %>% count(Type)

model.control <- rpart.control(minsplit = 3, xval = 10, cp = 0)
fit_wine <- rpart(Type~., data = train_data_cont, method = "class", control = model.control)

plot(fit_wine$cptable[,4], main = "Cp for model selection", ylab = "cv error")

min_cp = which.min(fit_wine$cptable[,4])
pruned_fit_wine <- prune(fit_wine, cp = fit_wine$cptable[min_cp,1])

rpart.plot(pruned_fit_wine,main="Pruned Tree")
text(pruned_fit_wine, use.n = T, all = T, cex = 1)

plot(pruned_fit_wine, branch = .3, compress=T, main = "Pruned Tree")
text(pruned_fit_wine, use.n = T, all = T, cex = 1)

rpart.plot(fit_wine,main="Full Tree")
text(fit_wine, use.n = T, all = T, cex = 1)

pred_train_full <- predict(fit_wine, newdata = train_data_cont, type = "class")
pred_test_full = predict(fit_wine, newdata = test_data_cont, type = "class")
confusionMatrix(pred_train_full, train_data_cont$Type)
confusionMatrix(pred_test_full, test_data_cont$Type)

pred_train_prune <- predict(pruned_fit_wine, newdata = train_data_cont, type = "class")
pred_test_prune = predict(pruned_fit_wine, newdata = test_data_cont, type = "class")
confusionMatrix(pred_train_prune, train_data_cont$Type)
confusionMatrix(pred_test_prune, test_data_cont$Type)
