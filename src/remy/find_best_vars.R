rm(list=ls())
set.seed(1)
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

library(InformationValue)
library(MLmetrics)
library(MASS)

write_csv <- function(pred_probas, filename="outputs/output.csv"){
  with_ind <- cbind(id=seq.int(length(pred_probas)), prob=pred_probas)
  probas <- as.data.frame(with_ind)
  write.csv(probas, file=filename, row.names = FALSE)
}

people <- read.csv(file="../../data/Dtrain.csv", header=TRUE)
test <- read.csv(file="../../data/Xtest1.csv", header=TRUE)

# Pre processing
purge_data <- function(data, vars){
  data$default <- NULL 
  contain_unknown <- apply(data, 1, function(r) any(r == "unknown"))
  restricted <- droplevels(data[!contain_unknown, ])
  return(restricted[, vars])
}

# Fitting to model
show_modinfos <- function(model){
  print(summary(model))
  print(anova(model, test='Chisq'))
}

stepwise_slct <- function(model, both_indep=FALSE, printit=FALSE){
  if(printit){
    print("Stepwise BOTH DIRECTIONS:")
  }
  step_both <- stepAIC(model, trace = FALSE)
  if(printit)
    print(step_both$anova)
  if(both_indep){
    print("Stepwise FORWARD")
    step_fw <- stepAIC(model, direction="forward", trace = FALSE)
    print(step_fw$anova)
    print("Stepwise BACKWARD")
    step_bw <- stepAIC(model, direction="backward", trace = FALSE)
    print(step_bw$anova)
    return(list(both=step_both, forward=step_fw, backward=step_bw))
  }
  return(list(both=step_both))
}

# Predictions 
new_predict <- function(model, obs=train_data){
  print("Calculating new predictions ...")
  preds <- predict(model, obs, type="response")
  print("Stats predictions:")
  print(summary(preds))
  return(preds)
}

cutoff <- function(y_obs, y_preds){
  opt <- optimalCutoff(y_obs, y_preds)[1]
  missed <- misClassError(y_obs, y_preds, threshold = opt)
  print(paste("Calculated threshold = ", opt, " that gives a missclass prop of ", missed))
  print("Confusion matrix with col=actual obs and row=predicted")
  print(confusionMatrix(y_obs, y_preds, threshold = opt))
  return(opt)
}

show_logloss <- function(y_actual, y_pred_prob, txt_for="model"){
  loss <- LogLoss(y_pred_prob, y_actual)
  print(paste("Log loss for ", txt_for, " = ", loss))
  return(loss)
}

plot_ROC_curves <- function(y_obs, y_preds){
  plotROC(y_obs, y_preds)
}

# Work

sep_dataset <- function(dataset, prop=5){
  obs_ok <- dataset[which(dataset$y == 1), ]
  obs_no <- dataset[which(dataset$y == 0), ]
  ok_for_valid <- sample(1:nrow(obs_ok), nrow(obs_ok)/prop)
  no_for_valid <- sample(1:nrow(obs_no), length(ok_for_valid))
  ind_obs_valid <- c(ok_for_valid, no_for_valid)
  print(paste("Dataset separation, obs for validation : ", length(no_for_valid), "with y=0 and", length(ok_for_valid), " y=1"))
  return(list(train=dataset[-ind_obs_valid, ], validation=dataset[ind_obs_valid, ]))
}

try_vars <- function(vars, train_set, valid_set, preprocess_fct=purge_data){
  infos <- list()
  infos$vars <- vars
  train_data <- if (is.null(preprocess_fct)) train_set[, c(vars, 'y')] else purge_data(train_set, c(vars, 'y'))
  model <- glm(y ~ ., family = 'binomial', data = train_data)
  infos$model <- model
  infos$model.anova <- anova(model, test='Chisq')
  valid_data <- if (is.null(preprocess_fct)) valid_set else purge_data(valid_set, c(vars, 'y'))
  
  train_pred <- predict(model, train_data, type="response")
  infos$train_pred <- train_pred
  infos$train_pred.confusion <- confusionMatrix(train_data$y, train_pred)
  infos$train_pred.loss <- LogLoss(train_pred, train_data$y)
  infos$train_pred.miss_ok <- infos$train_pred.confusion[1,2] / sum(infos$train_pred.confusion[, 2])
  
  valid_pred <- predict(model, valid_data, type="response")
  infos$valid_pred <- valid_pred
  infos$valid_pred.confusion <- confusionMatrix(valid_data$y, valid_pred)
  infos$valid_pred.loss <- LogLoss(valid_pred, valid_data$y)
  infos$valid_pred.miss_ok <- infos$valid_pred.confusion[1,2] / sum(infos$valid_pred.confusion[, 2])
  
  return(infos)
}

print_infos <- function(infos){
  print("Model fitting on variables :")
  print(infos$vars)
  print(summary(infos$model))
  print(infos$model.anova)
  stepwise <- stepwise_slct(infos$model)
  print("Stepwise in both dir resulted on :")
  print(stepwise$both$anova)

  print(paste("Predictions on TRAINING give logloss :", infos$train_pred.loss, "and prop y wrongly predicted as 0 :", infos$train_pred.miss_ok))
  print(infos$train_pred.confusion)

  print(paste("Predictions on VALIDATION give logloss :", infos$valid_pred.loss, "and prop y wrongly predicted as 0 :", infos$valid_pred.miss_ok))
  print(infos$valid_pred.confusion)
}

find_best <- function(vars, nbr, train_set, valid_set){
  poss_subsets <- combn(vars, nbr)
  res <- rep(NA, ncol(poss_subsets))
  for(i in 1:ncol(poss_subsets)){
    to_try <- poss_subsets[, i]
    try_res <- try_vars(c(to_try), train_set, valid_set, preprocess_fct = NULL)
    conf <- try_res$valid_pred.confusion
    res[i] <- conf
  }
  return(res)
}

apply_procedure <- function(vars, train_set, valid_set, test_set, preprocess_fct=NULL, printit=TRUE, plotit=TRUE, write_pred_probas=FALSE){
  infos <- try_vars(vars, train_set, valid_set, preprocess_fct)
  if (printit)
    print_infos(infos)
  final_preds <- predict(infos$model, test_set, type="response")
  print("Summary stats of fresh new TEST predictions :")
  print(summary(final_preds))
  if (plotit){
    h_train <- hist(infos$train_pred)
    h_valid <- hist(infos$valid_pred)
    h_test <- hist(final_preds)
    plot( h_train, col=rgb(0,0,1))
    plot( h_valid, col=rgb(1,0,0))
    plot( h_test, col=rgb(0,1,0))
    boxplot(infos$train_pred, infos$valid_pred, final_preds,
            names=c(paste('Training:', signif(infos$train_pred.miss_ok, 5)), paste('Validation:', signif(infos$valid_pred.miss_ok, 5)), 'Test'),
            horizontal = FALSE, col=c('blue', 'red', 'green'), cex=0.5, pch=20,
            main="Predictions on Y for different datasets with prop y=1 wrongly predicted")
  }
  if (write_pred_probas){
    write_csv(preds, filename = "outputs/procedure.csv")
  }
  return(list(infos=infos, test_preds=final_preds))
}

compute_test <- function(model){
  preds <- predict(model, test, type="response")
  write_csv(preds, filename = "outputs/adding_cst.csv")
}

vars = c('contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
sep <- sep_dataset(people)
infos <- try_vars(vars, sep$train, sep$validation, preprocess_fct = NULL)
apply_procedure(vars, sep$train, sep$validation, test)

# print_infos(infos)
# best <- find_best(vars, 5, sep$train, sep$validation)

#compute_test(glm(y ~ ., family = 'binomial', data = people[, c(vars, 'y')]))

# vars = c('age', 'job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
# 0.2545

# vars = c('age', 'job', 'marital', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome')
# 0.254469

# vars = c('age', 'job', 'marital', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'edu')
# 0.254307

# vars = c('age', 'job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'edu')
# 
# train_data <- purge_data(people, c(vars, 'y'))
# 
# print("######## Applying on real test set (unknown y) #########")
# print("## using complete training set ###")
# model <- glm(y ~ ., family = 'binomial', data = train_data)
# show_modinfos(model)
# 
# train_pred <- new_predict(model, obs = train_data)
# cut <- cutoff(train_data$y, train_pred)
# loss <- show_logloss(train_data$y, train_pred)
# stepwise <- stepwise_slct(model)$both
