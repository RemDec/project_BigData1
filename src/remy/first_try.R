rm(list=ls())
set.seed(1)
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

library(InformationValue)

people <- read.csv(file="../../data/Dtrain.csv", header=TRUE)
test <- read.csv(file="../../data/Xtest1.csv", header=TRUE)

# Pre-processing
#people$subs = as.factor(people$y)

print(table(people$y))

pred_ok <- people[which(people$y == 1), ]
pred_no <- people[which(people$y == 0), ]

training_one <- sample(1:nrow(pred_ok), 0.8*nrow(pred_ok))
training_zero <- sample(1:nrow(pred_ok), 0.8*nrow(pred_ok))
training_ind = c(training_one, training_zero)

train_set <- people[training_ind, ]
test_set <- people[-training_ind, ]

lrfit <- glm(y ~ marital + edu + housing + previous + month, data=people, family='binomial')

print(summary(lrfit))
print(anova(lrfit, test='Chisq'))

pred <- predict(lrfit, test_set, type='response')
cutoff <- optimalCutoff(test_set$y, pred)
predict_res <- ifelse(pred > 0.5, 1, 0)
miss <- mean(pred != test_set$y)
print(paste('accuracy : ', 1-miss))



