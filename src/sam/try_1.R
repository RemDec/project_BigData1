library(MASS)
library(InformationValue)
library(MLmetrics)
library(rpart)

set.seed(100)

#Rémy function
write_csv <- function(pred_probas, filename="outputs/output.csv"){
  with_ind <- cbind(id=seq.int(length(pred_probas)), prob=pred_probas)
  probas <- as.data.frame(with_ind)
  write.csv(probas, file=filename, row.names = FALSE)
}

#data sets loading

people <- read.csv(file="../../data/Dtrain.csv", header=TRUE)
test <- read.csv(file="../../data/Xtest1.csv", header=TRUE)

peopleRework <- people[!(people$marital=="unknown" & people$default=="unknown" & people$housing=="unknown" & people$loan=="unknown"),]

#data pre working

people.one <- people[which(people$y == 1),]
people.one.active <- people.one[-14]

#anal

#model.glm <- glm(y ~ age + job + marital + edu + previous + month + contact + pdays, family = binomial(link='logit'), data = people)
model.lda <- lda(y ~ age + job + marital + edu + previous + month + contact + pdays + poutcome, data = people)
model.lda2 <- lda(y ~ age + job + marital + edu + previous + month + contact + pdays + poutcome, data = peopleRework)

model.lda.predict <- predict(model.lda, test, type="response")
model.lda.predict2 <- predict(model.lda2, test, type="response")
model.lda.predictB <- predict(model.lda, people[-14], type="response")

LogLoss(model.lda.predictB$posterior[,2], people$y)

write_csv(model.lda.predict$posterior[,2], filename="outputs/sam_try2.csv")
hist(model.lda.predict$posterior[,2])
hist(model.lda.predict2$posterior[,2])
