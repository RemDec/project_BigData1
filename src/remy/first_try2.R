rm(list=ls())
set.seed(1)
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

library(InformationValue)

people <- read.csv(file="../../data/Dtrain.csv", header=TRUE)
test <- read.csv(file="../../data/Xtest1.csv", header=TRUE)

# Pre processing
people <- people[which(people$default != 'yes'), ]
people$default <- droplevels(people$default)

pred_ok <- people[which(people$y == 1), ]
pred_no <- people[which(people$y == 0), ]
pred_no_for_train <- sample(1:nrow(pred_no), nrow(pred_no)/3)

train_data <- people
train_data_sized <- rbind(pred_ok,pred_no[pred_no_for_train, ])
test_data_sized <- people[-pred_no_for_train, ]

model <- glm(y ~ ., family = 'binomial', data = train_data)

# Utilities functions

write_csv <- function(pred_probas, filename="outputs/output.csv"){
  with_ind <- cbind(id=seq.int(length(pred_probas)), prob=pred_probas)
  probas <- as.data.frame(with_ind)
  write.csv(probas, file=filename, row.names = FALSE)
}

show_modinfos <- function(model){
  print(summary(model))
  print(anova(model, test='Chisq'))
}

new_predict <- function(model, obs=train_data){
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

plot_ROC_curves <- function(y_obs, y_preds){
  plotROC(y_obs, y_preds)
}

# Work


print("######## Applying on real test set (unknown y) #########")
print("## using complete training set ###")
model <- glm(y ~ ., family = 'binomial', data = train_data)
show_modinfos(model)
test_pred <- new_predict(model, obs = test)
p1 <- hist(test_pred)

print("##")
print("## using reduced training set to prevent overfitting ###")
model <- glm(y ~ age + job + contact + month + day_of_week + campaign + pdays + previous, family = 'binomial', data = train_data_sized)


show_modinfos(model)
test_pred <- new_predict(model, obs = test)
p2 <- hist(test_pred)

test_pred_sized <- new_predict(model, obs = test)
p2 <- hist(test_pred_sized)

plot( p1, col=rgb(0,0,1))  # first histogram
plot( p2, col=rgb(1,0,0, 0.8), add=T)

write_csv(test_pred, filename = "outputs/first_try_nonsized.csv")
write_csv(test_pred_sized, filename = "outputs/first_try_resized.csv")