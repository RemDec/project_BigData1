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

compute_test <- function(model){
  preds <- predict(model, test, type="response")
  write_csv(preds, filename = "outputs/adding_cst.csv")
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

simplify_jobs <- function(data, vars){
  if ('job' %in% vars){
    replacement <- function(r){
      if (r['job'] %in% c('blue-collar', 'housemaid', 'services', 'technician', 'unknown'))
        return('poor')
      if (r['job'] %in% c('admin.', 'entrepreneur', 'management', 'self-employed', 'unemployed'))
        return('middle')
      if (r['job'] %in% c('retired', 'student'))
        return('good')
    }
    data$job <- as.factor(apply(data, 1, replacement))
  }
  return(data)
}

categoric_pdays <- function(data, vars){
  if ('pdays' %in% vars){
    replacement <- function(r){
      if (r['pdays'] == 999)
        return('never')
      if (r['pdays'] > 5)
        return('late')
      return('recent')
    }
    data$pdays <- as.factor(apply(data, 1, replacement))
  }
  return(data)
}

preprocess <- function(data, vars){
  data <- simplify_jobs(data, vars)
  data <- categoric_pdays(data, vars)
  return(data)
}

get_incoherent <- function(){
  return(which(people$pdays == 999 & people$previous > 0))
}

# Fitting to model
show_modinfos <- function(model){
  print(summary(model))
  print(anova(model, test='Chisq'))
}

stepwise_slct <- function(model, data, both_indep=FALSE, printit=FALSE){
  model$call$data <- 'data'
  #model$data <- data
  if(printit)
    print("Stepwise BOTH DIRECTIONS:")
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
  print("Stats on new predictions:")
  print(summary(preds))
  return(preds)
}

cutoff <- function(y_obs, y_preds, use_opt_thresh=FALSE){
  opt <- if (use_opt_thresh) optimalCutoff(y_obs, y_preds)[1] else 0.5
  missed <- misClassError(y_obs, y_preds, threshold = opt)
  print(paste("Calculated/given threshold = ", opt, " that gives a missclass prop of ", missed))
  print("Confusion matrix with col=actual obs and row=predicted")
  conf <- confusionMatrix(y_obs, y_preds, threshold = opt)
  missed_ok <- conf[1,2] / sum(conf[, 2])
  print(paste("Proportion of y missclassed as 0 =", missed_ok))
  print(conf)
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

sep_dataset <- function(dataset, prop=5, balance=0, printit=TRUE){
  ind_obs_ok <- which(dataset$y == 1)
  ind_obs_no <- which(dataset$y == 0)
  obs_ok <- dataset[ind_obs_ok, ]
  obs_no <- dataset[ind_obs_no, ]
  ok_for_valid <- sample(ind_obs_ok, nrow(obs_ok)/prop)
  no_for_valid <- sample(ind_obs_no, length(ok_for_valid))
  ind_obs_valid <- c(ok_for_valid, no_for_valid)
  if (printit)
    print(paste("Dataset separation, obs for validation : ", length(no_for_valid), "with y=0 and", length(ok_for_valid), " y=1",
                " TOTAL validation set size =", length(ind_obs_valid)))
  if (balance != 0){
    remaining <- dataset[-ind_obs_valid, ]
    ind_remaining_ok <- which(remaining$y == 1)
    ind_remaining_no <- which(remaining$y == 0)
    slcted_no <- sample(ind_remaining_no, min(length(ind_remaining_no), length(ind_remaining_ok)*balance))
    ind_obs_training <- c(ind_remaining_ok, slcted_no)
    train_data <- remaining[ind_obs_training, ]
    if (printit){
      print(paste("Nbr ok remaining", length(ind_remaining_ok)))
      print(paste("Nbr no remaining", length(ind_remaining_no)))
      print(paste("Nbr no selected in remaining (remaining ok x", balance, ") :", length(slcted_no)))
      print(paste("TOTAL training set size = (", length(ind_remaining_ok), "+", length(slcted_no), ") =", length(ind_obs_training)))
    }
  }else{
    train_data <- dataset[-ind_obs_valid, ]
  }
  validation_data <- dataset[ind_obs_valid, ]
  if (printit){
    print("Real repartition of y in TRAINING selected data")
    print(table(train_data$y))
    print("Real repartition of y in VALIDATING selected data")
    print(table(validation_data$y))
  }
  return(list(train=train_data, validation=validation_data))
}

try_vars <- function(vars, train_set, valid_set, preprocess_fct=NULL){
  infos <- list()
  infos$vars <- vars
  train_data <- if (is.null(preprocess_fct)) train_set[, c(vars, 'y')] else preprocess_fct(train_set, c(vars, 'y'))[, c(vars, 'y')]
  model <- glm(y ~ ., family = 'binomial', data = train_data)
  infos$model <- model
  infos$model.anova <- anova(model, test='Chisq')
  valid_data <- if (is.null(preprocess_fct)) valid_set else preprocess_fct(valid_set, c(vars, 'y'))
  # Care to prevent auto removed factor level at CV
  model$xlevels$month <- union(model$xlevels$month, c(levels(valid_data$month), levels(train_data$month)) )
  
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
  # try(stepwise <- stepwise_slct(infos$model, printit = TRUE))

  print(paste("Predictions on TRAINING give logloss :", infos$train_pred.loss, "and prop y wrongly predicted as 0 :", infos$train_pred.miss_ok))
  print(infos$train_pred.confusion)

  print(paste("Predictions on VALIDATION give logloss :", infos$valid_pred.loss, "and prop y wrongly predicted as 0 :", infos$valid_pred.miss_ok))
  print(infos$valid_pred.confusion)
}


apply_procedure <- function(vars, train_set, valid_set, test_set, preprocess_fct=NULL, printit=TRUE, plotit=TRUE, write_pred_probas=FALSE, filename='outputs/last_predictions.csv'){
  infos <- try_vars(vars, train_set, valid_set, preprocess_fct)
  if (printit)
    print_infos(infos)
  test_set <- if (is.null(preprocess_fct)) test_set else preprocess_fct(test_set, c(vars, 'y'))
  infos$model$xlevels$month <- union(infos$model$xlevels$month, levels(test_set$month))
  final_preds <- predict(infos$model, test_set, type="response")
  prop_test_ok <- sum(final_preds > 0.5) / length(final_preds)
  if (printit){
    print("Summary stats of fresh new TEST predictions :")
    print(summary(final_preds))
  }
  if (plotit){
    h_train <- hist(infos$train_pred)
    h_valid <- hist(infos$valid_pred)
    h_test <- hist(final_preds)
    plot( h_train, col=rgb(0,0,1))
    plot( h_valid, col=rgb(1,0,0))
    plot( h_test, col=rgb(0,1,0))
    n <- c(paste('Training:', signif(infos$train_pred.miss_ok, 5)),
           paste('Validation:', signif(infos$valid_pred.miss_ok, 5)),
           paste('Test: prop > 0.5 =', signif(prop_test_ok, 5)))
    boxplot(infos$train_pred, infos$valid_pred, final_preds,
            names=n,
            horizontal = FALSE, col=c('blue', 'red', 'green'), cex=0.5, pch=20,
            main="Predictions on Y for different datasets with prop y=1 wrongly predicted")
  }
  if (write_pred_probas)
    write_csv(final_preds, filename = filename)
  return(list(infos=infos, test_preds=final_preds))
}


apply_CV_procedure <- function(vars, obs_set, nbr_folds=5, preprocess_fct=NULL){
  folds <- cut(sample(seq(1, nrow(obs_set))), breaks = nbr_folds, labels=FALSE)
  CV_infos <- list()
  for(i in 1:nbr_folds){
    # dividing observations set into training and validation test set
    test_inds <- which(folds==i, arr.ind=TRUE)
    test_set <- obs_set[test_inds, ]
    train_set <- obs_set[-test_inds, ]
    # applying model (Log. Regr. with glm) and computing measurements on it
    sep <- sep_dataset(train_set, prop = 10, balance = 5, printit = FALSE)
    infos <- try_vars(vars, sep$train, test_set, preprocess_fct=preprocess_fct)
    CV_infos[[i]] <- infos
    print(paste("   |_ Applying procedure considering fold", i,
                "(sizes: train=", nrow(sep$train), ", test=", nrow(test_set), ")"))
    print(paste("     | Validation logloss=", infos$valid_pred.loss))
    print(paste("     | Misspredicted y=1 as 0 :", infos$valid_pred.miss_ok))
  }
  return(CV_infos)
}

interpret_CV_infos <- function(CV_infos){
  folds <- length(CV_infos)
  losses <- rep(NA, folds)
  train_misses <- rep(NA, folds)
  misses <- rep(NA, folds)
  AICs <- rep(NA, folds)
  for(i in 1:folds){
    infos <- CV_infos[[i]]
    losses[i] <- infos$valid_pred.loss
    train_misses[i] <- infos$train_pred.miss_ok
    misses[i] <- infos$valid_pred.miss_ok
    AICs[i] <- infos$model$aic
  }
  return(list(loglosses=losses, train_misses_ok=train_misses, misses_ok=misses, aics=AICs))
}

generate_CVs <- function(vars, obs_set, nbr_CV=15, nbr_folds_per_CV=10, preprocess_fct=NULL){
  mean_losses <- rep(NA, nbr_CV)
  mean_train_misses <- rep(NA, nbr_CV)
  mean_misses <- rep(NA, nbr_CV)
  mean_aics <- rep(NA, nbr_CV)
  for(i in 1:nbr_CV){
    print(" ----------------------------------------")
    print(paste("Launching Cross-Validation number ", i))
    CV_infos <- apply_CV_procedure(vars, obs_set, nbr_folds_per_CV, preprocess_fct)
    CV_stats <- interpret_CV_infos(CV_infos)
    mean_losses[i]  <- mean(CV_stats$loglosses)
    mean_train_misses[i] <- mean(CV_stats$train_misses_ok) 
    mean_misses[i] <- mean(CV_stats$misses_ok)
    mean_aics[i] <- mean(CV_stats$aics)
  }
  boxplot(mean_losses, main="Stats on means log losses", outline = FALSE)
  boxplot(mean_misses, mean_train_misses, col=c("green", "orange"), names=c("Validation", "Training"),
          main="Stats on means proportion of y=1 wrongly predicted as 0")
  boxplot(mean_aics, main="Stats on means AIC of models")
  print("CrossValidation with variables :")
  print(vars)
  print(paste("Mean of mean losses =", mean(mean_losses)))
  print(paste("Mean of mean misses : for training =", mean(mean_train_misses), ", for validation =", mean(mean_misses)))
  print(paste("Mean of mean AICs =", mean(mean_aics)))
}

simulate_test <- function(model, nbr_obs=10182, indata=people, preprocess_fct=NULL, use_opt_thresh=FALSE){
  print("------------------------------------------")
  print(paste("SIMULATING TEST picking", nbr_obs, "observations"))
  slcted_data <- indata[sample(1:nrow(indata), nbr_obs), ]
  slcted_data <- if (is.null(preprocess_fct)) slcted_data else preprocess_fct(slcted_data, colnames(slcted_data))
  preds <- new_predict(model, slcted_data)
  cutoff(slcted_data$y, preds, use_opt_thresh = use_opt_thresh)
  show_logloss(slcted_data$y, preds)
  plot_ROC_curves(slcted_data$y, preds)
}

plot_against_best <- function(new_preds){
  h1 <- hist(new_preds)
  best <- read.csv(file="../../data/best_entry.csv", header=TRUE)
  h2 <- hist(best[, 'prob'])
  plot(h1, col=rgb(0,0,1))  # first histogram
  plot(h2, col=rgb(1,0,0, 0.8), add=T)
}

plot_logloss_from_balance <- function(vars, range=0:10, prop=10, nbr_model=4, preprocess_fct=NULL){
  mean_logs_valid <- rep(NA, length(range))
  mean_logs_train <- rep(NA, length(range))
  test_predictions <- list()
  ind <- 1
  for(i in range){
    logs_valid <- rep(NA, nbr_model)
    logs_train <- rep(NA, nbr_model)
    for (mod_inst in 1:nbr_model){
      sep <- sep_dataset(people, prop = prop, balance = i, printit = FALSE)
      results <- apply_procedure(vars, sep$train, sep$validation, test, preprocess_fct=preprocess_fct, printit = FALSE, plotit = FALSE)
      logs_valid[mod_inst] <- results$infos$valid_pred.loss
      logs_train[mod_inst] <- results$infos$train_pred.loss
      preds <- results$test_preds
      nbr_obs_valid <- nrow(sep$validation)
    }
    mean_logs_valid[ind] <- mean(logs_valid)
    mean_logs_train[ind] <- mean(logs_train)
    test_predictions[[ind]] <- preds
    ind <- ind + 1
  }
  maxloss <- max(c(mean_logs_train, mean_logs_valid))
  plot(range, mean_logs_valid, col=rgb(0,1,0), type="o", ylim=c(0.15, maxloss),
       xlab="Balance parameter value", ylab="Mean logloss value", 
       main=paste("Logloss variation considering set separation balance (", nbr_obs_valid,"obs in val. set)"))
  lines(range, mean_logs_train, col=rgb(1,0,0), type="o", ylim=c(0.15, maxloss))
  legend(x="top", legend = c("validation", "training"), col = c("green", "red"), cex=0.8, lty=1:2)
  boxplot(test_predictions, cex=0.5, pch=20,
          main="Predictions distributions for test set considering balance param.", names = paste("Bal=", range))
}


best_result <- function(preprocess_fct=NULL){
  vars = c('job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
  sep <- sep_dataset(people, prop = 10, balance = 5)
  results <- apply_procedure(vars, rbind(sep$train, sep$validation), sep$validation, test, preprocess_fct=preprocess_fct, printit = FALSE, write_pred_probas = TRUE, filename="outputs/new_best_preds.csv")
  print_infos(results$infos)
  plot_against_best(results$test_preds)
  simulate_test(results$infos$model, nbr_obs = nrow(people), preprocess_fct = preprocess_fct, use_opt_thresh = FALSE)
  generate_CVs(vars, people, nbr_CV = 5, nbr_folds_per_CV = 5, preprocess_fct = preprocess_fct)
  hist(results$test_preds, col="green", breaks = 40,
       main="Model : test predictions probabilities", xlab="Predicted probability values")
}

vars_indiv <- c('age', 'job', 'marital', 'housing', 'loan')
vars_camp <- c('contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome')

vars = c('age', 'job', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
sep <- sep_dataset(people, prop = 10, balance = 6)
results <- apply_procedure(vars, rbind(sep$train, sep$validation), sep$validation, test, preprocess_fct=NULL, printit = FALSE, write_pred_probas = TRUE)
print_infos(results$infos)

plot_against_best(results$test_preds)
simulate_test(results$infos$model, nbr_obs = nrow(people), preprocess_fct = NULL, use_opt_thresh = FALSE)
generate_CVs(vars, people, nbr_CV = 5, nbr_folds_per_CV = 4, preprocess_fct = preprocess)


# # Best false negative rate 0.90
#
# vars = c('job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
# sep <- sep_dataset(people, prop = 10, balance = 5)
# results <- apply_procedure(vars, sep$train, people, test, preprocess_fct=preprocess, printit = FALSE, write_pred_probas = TRUE)
# print_infos(results$infos)
# print(summary(results$test_preds))
# 
# plot_against_best(results$test_preds)



# # Versus Best
# vars = c('job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
# sep <- sep_dataset(people, prop = 5)
# 
# # simulate_test(vars, people, preprocess_fct = preprocess)
# results <- apply_procedure(vars, people, sep$validation, test, preprocess_fct=preprocess, printit = FALSE, write_pred_probas = TRUE)
# print_infos(results$infos)
# print(summary(results$test_preds))
# 
# plot_against_best(results$test_preds)

# # Good
# vars = c('age', 'job', 'marital', 'default', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'edu')
# sep <- sep_dataset(people, prop = 6)
# 
# results <- apply_procedure(vars, sep$train, sep$validation, test, printit = FALSE)
# print_infos(results$infos)

# # Better
# vars = c('job', 'marital', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous')
# sep <- sep_dataset(people, prop = 6)
# 
# results <- apply_procedure(vars, sep$train, sep$validation, test, printit = FALSE)
# print_infos(results$infos)
