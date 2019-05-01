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

plot_curves <- function(y_obs, y_preds){
  plotROC(y_obs, y_preds)
}

# Work

show_modinfos(model)
#preds <- new_predict(obs=test_data)

#thresh <- cutoff(test_data$y, preds)
#plot_curves(test_data$y, preds)

print("######## Applying on real test set (unknown y) #########")
print("## using complete training set ###")
test_pred <- new_predict(model, obs = test)
p1 <- hist(test_pred)

print("## using reduced training set to prevent overfitting ###")
model <- glm(y ~ ., family = 'binomial', data = train_data_sized)
test_pred <- new_predict(model, obs = test)
p2 <- hist(test_pred)

plot( p1, col=rgb(0,0,1))  # first histogram
plot( p2, col=rgb(1,0,0, 0.8), add=T)