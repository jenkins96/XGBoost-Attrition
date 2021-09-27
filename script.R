# Libraries ---------------------------------------------------------------
listOfPackages <- c("tidyverse", "magrittr", "xgboost", 
                    "caret", "tidymodels", "DiagrammeR",
                    "doParallel")

for (i in listOfPackages){
  if(! i %in% installed.packages()){
    install.packages(i, dependencies = TRUE)
  }
  lapply(i, require, character.only = TRUE)
  rm(i)
}


# Importing Data ----------------------------------------------------------

data <- read_csv("dataset/XGBoost.csv")
attrition <- data

attrition %<>%
  select(c("Age", "Attrition_XGBoost",
           "BusinessTravel", "Education",
           "JobLevel", "MonthlyIncome",
           "OverTime")) %>%
  rename("Attrition" = "Attrition_XGBoost")


str(attrition)



# Dataset Partitioning ----------------------------------------------------
set.seed(1)
attrition_split <- initial_split(attrition, strata = Attrition)
attrition_train <- training(attrition_split)
attrition_test <- testing(attrition_split)
attrition_split


# Dummy Variables ---------------------------------------------------------

trainY <-  attrition_train$Attrition == "1"
trainX <- model.matrix(Attrition ~.-1, data = attrition_train)



testY <- attrition_test$Attrition == "1"
testX <- model.matrix(Attrition ~. -1, data = attrition_test)


# DMatrix
Xmatrix_training <-  xgb.DMatrix(data = trainX, label = trainY)
Xmatrix_testing <-xgb.DMatrix(data = testX, label = testY)


# XGBoosting Model --------------------------------------------------------


Xgboosting <- xgboost(data = Xmatrix_training,
                      objective = "multi:softmax",
                      num_class = 2,
                      nrounds = 51,
                      eval_metric="merror"
)

# Predicting
xgpred <- predict(Xgboosting, Xmatrix_testing)
my_table <- table(testY, xgpred, dnn = c("Actual", "Predicted"))

# Metrics
TP <- my_table [2,2]
TN <- my_table [1,1]
FP <- my_table [1,2]
FN <- my_table [2,1]
# Accuracy
accuracy <- sum(TP, TN)/sum(TP,FP,FN,TN)
sprintf("Accuracy: %f", accuracy)
# Recall
recall<- TP/(sum(TP, FN)) 
sprintf("Recall: %f", recall[1])
# Specificity
specificity <- TN/(sum(TN,FP))
sprintf("Specificity: %f", specificity[1])

# Precision
precision <-  TP/ sum(TP,FP)
sprintf("Precision: %f", precision)
# F1
F1 <- 2*(recall * precision)/(recall+precision)
sprintf("F1 Score: %f", F1)

err <- mean(as.numeric(xgpred > 0.5) != testY)



# Optimizing Model --------------------------------------------------------

cl <- makePSOCKcluster(9) # I have found this number is optimal for my pc. 
registerDoParallel(cl) # You could also run it without doParallel library, just will take longer

# Take start time to measure time of random search algorithm
start.time <- proc.time()

# Create empty lists
lowest_merror_list = list()
parameters_list = list()

# Create 10,000 rows with random hyperparameters

set.seed(20)
for (iter in 1:10000){
  param <- list(booster = "gbtree",
                objective = "multi:softmax",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .7, 1),
                colsample_bytree = runif(1, .6, 1),
                min_child_weight = sample(0:10, 1),
                gamma = floor(runif(1, 0, 100)),
                alpha = runif(1, 0, 100),
                lambda = runif(1, 0, 100)
                
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}

# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

# Use randomly created parameters to create 10,000 XGBoost-models

for (row in 1:nrow(parameters_df)){
  set.seed(20)
  mdcv <- xgb.train(data=Xmatrix_training,
                    booster = "gbtree",
                    objective = "multi:softmax",
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    gamma = parameters_df$gamma[row],
                    alpha = parameters_df$alpha[row],
                    lambda = parameters_df$lambda[row],
                    nrounds= 351,
                    eval_metric = "merror",
                    early_stopping_rounds= 30,
                    print_every_n = 100,
                    num_class = 2,
                    watchlist = list(train= Xmatrix_training, val= Xmatrix_testing)
  )
  lowest_merror <-  as.data.frame(1 - min(mdcv$evaluation_log$val_merror))
  lowest_merror_list[[row]] = lowest_merror
}

# Create object that contains all accuracy's
lowest_merror_df = do.call(rbind, lowest_merror_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = cbind(lowest_merror_df, parameters_df)

# Quickly display highest accuracy
maxacc <- max(randomsearch$`1 - min(mdcv$evaluation_log$val_merror)`)
print(randomsearch %>% filter(`1 - min(mdcv$evaluation_log$val_merror)` == maxacc))

# Stop time and calculate difference
end.time <- proc.time()
time.taken <- end.time - start.time
print(time.taken)

write_csv(randomsearch, "dataset/randomsearch.csv")

stopCluster(cl)


## Best Parameters 
# 0.859079
# max_depth = 10
# eta = 0.2276653
# subsample = 0.8562708
# colsample_bytree = 0.9899229
# min_child_weight = 5
# gamma = 0
# alpha = 0.04643467
# lambda = 0.8723213



# Using tuned params

Xgboosting2 <- xgboost(data = Xmatrix_training,
                       objective = "multi:softmax",
                       num_class = 2,
                       nrounds = 251,
                       eval_metric = "merror",
                       max_depth = 10,
                       eta = 0.2276,
                       subsample = 0.8562,
                       colsample_bytree = 0.9899,
                       min_child_weight = 5,
                       gamma = 0,
                       alpha = 0.04643,
                       lambda = 0.8723,
                       set.seed(1)
)


# Predicting
xgpred2 <- predict(Xgboosting2, Xmatrix_testing)
my_table2 <- table(testY, xgpred2, dnn = c("Actual", "Predicted"))


# Metrics
TP_2 <- my_table2 [2,2]
TN_2 <- my_table2 [1,1]
FP_2 <- my_table2 [1,2]
FN_2 <- my_table2 [2,1]

# Accuracy
accuracy_2 <- sum(TP_2, TN_2)/sum(TP_2,FP_2,FN_2,TN_2)
sprintf("Accuracy: %f", accuracy_2)
# Recall
recall_2<- TP_2/(sum(TP_2, FN_2)) 
sprintf("Recall: %f", recall_2)
# Specificity
specificity_2 <- TN_2/(sum(TN_2,FP_2))
sprintf("Specificity: %f", specificity_2)

# Precision
precision_2 <-  TP_2/ sum(TP_2,FP_2)
sprintf("Precision: %f", precision_2)
# F1
F1_2 <- 2*(recall_2 * precision_2)/(recall_2+precision_2)
sprintf("F1 Score: %f", F1_2)

err_2 <- mean(as.numeric(xgpred2 > 0.5) != testY)


# Feature Importance
importance_matrix <- xgb.importance(model = Xgboosting2)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix[1:3,], 
                    main = "Feature Importance \n Top 3",
                    xlab = "Gain"
)
