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
  
  #levels(restricted$job)[which(levels(restricted$job) %in% c("blue-collar", "housemaid", "student", "unemployed"))] <- "poor"
  #levels(restricted$job)[which(levels(restricted$job) %in% c("entrepreneur", "management", "self-employed", "services", "technician"))] <- "middle"
  return(restricted[, vars])
}

# Fitting to model
show_modinfos <- function(model){
  print(summary(model))
  print(anova(model, test='Chisq'))
}

stepwise_slct <- function(model, both_indep=FALSE){
  print("Stepwise BOTH DIRECTIONS:")
  step_both <- stepAIC(model, trace = FALSE)
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

# vars = c('age', 'job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
# 0.2545

# vars = c('age', 'job', 'marital', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome')
# 0.254469

# vars = c('age', 'job', 'marital', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'edu')
# 0.254307

vars = c('age', 'job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'edu')

train_data <- purge_data(people, c(vars, 'y'))

print("######## Applying on real test set (unknown y) #########")
print("## using complete training set ###")
model <- glm(y ~ ., family = 'binomial', data = train_data)
show_modinfos(model)

train_pred <- new_predict(model, obs = train_data)
cut <- cutoff(train_data$y, train_pred)
loss <- show_logloss(train_data$y, train_pred)
stepwise <- stepwise_slct(model)$both


# library(glmnet)
# grid <-  10^seq(10, -2, length=100)
# ridge.model <-glmnet(train_data, train_data$y, lambda = grid, alpha = 0, family = "binomial")
# lasso.model <-glmnet(train_data, train_data$y, lambda = grid, alpha = 1, family = "binomial")

#plot_ROC_curves(train_data$y, train_pred)

# Predict test set
# purged_test <- purge_data(test, vars)
# test_pred <- new_predict(model, obs = purged_test)
# p1 <- hist(test_pred)
# plot( p1, col=rgb(0,0,1))
# write_csv(test_pred, filename = "outputs/pred_without_unknown.csv")