library(neuralnet)

setwd("/Users/taoni/Desktop/COMP8410/Assignment_2/")

poll <- read.csv("ANUPoll2018Data_CSV_01428.csv")

data <- subset(poll, select = c("Q8b", "Q8c", "Q8d", "Q8e", "Q8f", "Q8g", "Q8h", "Q8a"))

data[is.na(data)] <- 0

index <- sample(1:nrow(data), round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]

f <- "Q8a ~ Q8b + Q8c + Q8d + Q8e + Q8f + Q8g + Q8h"
nn <- neuralnet(f, data=train_, hidden=c(5,3), act.fct = "logistic", linear.output = T)
plot(nn)

pr.nn <- compute(nn, test_[,1:7])
pr.nn_ <- pr.nn$net.result*(max(data$Q8a)-min(data$Q8a)) + min(data$Q8a)
test.r <- (test_$Q8a)*(max(data$Q8a)-min(data$Q8a)) + min(data$Q8a)

plot(test$Q8a, pr.nn_, col='red', main = 'Real vs predicted NN', pch=18, cex=0.7)

MAE.nn <- sum(abs(test.r - pr.nn_))/nrow(test_)
print(MAE.nn)

lm.fit <- glm(f, data=train)
pr.lm <- predict(lm.fit, test)
summary(lm.fit)

MAE.lm <- sum(abs(pr.lm - test$Q8a))/nrow(test)
print(MAE.lm)

par(mfrow=c(1,1))

plot(test$Q8a, pr.nn_, col='red', main="Real vs predicted NN", pch=18, cex=0.7)
abline(0, 1, lwd=2)
legend('bottomright', legend='NN', pch=18, col='red', bty='n')

plot(test$Q8a, pr.lm, col='blue', main="Real vs predicted lm", pch=18, cex=0.7)
abline(0, 1, lwd=2)
legend('bottomright', legend='LM', pch=18, col='blue', bty='n', cex=0.95)
