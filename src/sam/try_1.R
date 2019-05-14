library(MASS)
library(InformationValue)
library(MLmetrics)
library(rpart)

set.seed(100)

#R?my function
write_csv <- function(pred_probas, filename="outputs/output.csv"){
  with_ind <- cbind(id=seq.int(length(pred_probas)), prob=pred_probas)
  probas <- as.data.frame(with_ind)
  write.csv(probas, file=filename, row.names = FALSE)
}

#sam functions
process_loan <- function(data){
  replacement <- function(r){
    l = c(r['default'], r['housing'], r['loan'])
    if ('yes' %in% l)
      return('yes')
    if ('unknown' %in% l)
      return('unknown')
    return('no')
  }
  data$default <- as.factor(apply(data, 1, replacement))
  return(data[-(5:6)])
}

#data sets loading

people <- read.csv(file="../../data/Dtrain.csv", header=TRUE)
test <- read.csv(file="../../data/Xtest1.csv", header=TRUE)

peopleRework <- process_loan(people)

#data pre working

people.one <- people[which(people$y == 1),]
people.one.active <- people.one[-14]

#anal

#model.glm <- glm(y ~ age + job + marital + edu + previous + month + contact + pdays, family = binomial(link='logit'), data = people)
model.lda <- lda(y ~ age + job + marital + edu + previous + month + contact + pdays + poutcome, data = people)
model.lda2 <- lda(y ~ age + job + marital + edu + previous + month + contact + pdays + poutcome + default, data = peopleRework)

model.lda.predict <- predict(model.lda, test, type="response")
model.lda.predict2 <- predict(model.lda2, test, type="response")
model.lda.predictB <- predict(model.lda, people[-14], type="response")
model.lda.predictBRe <- predict(model.lda2, peopleRework[-12], type="response")

LogLoss(model.lda.predictB$posterior[,2], people$y)

write_csv(model.lda.predict$posterior[,2], filename="outputs/sam_try2.csv")
h <- hist(model.lda.predict$posterior[,2], col="red", breaks=20)
text(h$mids, h$counts, labels=h$counts, adj=c(0.5, -0.5))
hist(model.lda.predict2$posterior[,2])

h <- hist(model.lda.predictB$posterior[,2], col="green", breaks=20)
text(h$mids, h$counts, labels=h$counts, adj=c(0.5, -0.5))
h <- hist(model.lda.predictBRe$posterior[,2], col="blue", breaks=20)
text(h$mids, h$counts, labels=h$counts, adj=c(0.5, -0.5))
