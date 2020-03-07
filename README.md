# Data-Mining-for-Reporting-the-Results-of-Monte-Carlo-Simulations
Code from "Data Mining for Reporting the Results of Monte Carlo Simulations"


#read in tree data
source(".trees.R")

#make outcome is numeric
tree.data$BIC <- ifelse(tree.data$BIC == "Yes", 1, 0)
tree.data$Entropy <- ifelse(tree.data$Entropy == "Yes", 1, 0)


tree.data <- transform(
  tree.data,
  Method=as.factor(Method),
  Sample.Size=as.factor(Sample.Size),
  M.Distance=as.factor(M.Distance),
  Iter=as.integer(Iter),
  BIC=as.factor(BIC),
  Num.of.Classes=as.factor(Num.of.Classes),
  Entropy=as.factor(Entropy)
)

#############################Classification Trees
#split the data
## 80% of the sample size
smp_size <- floor(0.80 * nrow(tree.data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(tree.data)), size = smp_size)

train <- tree.data[train_ind, ]
test <- tree.data[-train_ind, ]



tree.BIC = tree(BIC~Method + Sample.Size + M.Distance , data=train)


tree.pred = predict(tree.BIC, test, type="class")

#Then you can evalute the error by using a misclassification table.
with(test, table(tree.pred, BIC))


tree.Entropy = tree(Entropy~Method + Sample.Size + M.Distance , data=train)
tree.pred = predict(tree.Entropy, test, type="class")
#Then you can evalute the error by using a misclassification table.
with(test, table(tree.pred, Entropy))




#Let's see the summary of your classification tree:

summary(tree.pred)

#plot
plot(tree.pred)


#############################Random Forest

#read in the library to conduct random forest algorithm
library(randomForest)

#split the data
## 80% of the sample size
smp_size <- floor(0.80 * nrow(tree.data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(tree.data)), size = smp_size)

train <- tree.data[train_ind, ]
test <- tree.data[-train_ind, ]

#train our random forest model
(rf <- randomForest(
  BIC ~ Method + Sample.Size + M.Distance ,
  data=train, n.trees=501
))

plot(rf)


#we use our model to predict the testing data
pred = predict(rf, newdata=test[,c("Method", "Sample.Size",
                                  "M.Distance",  "E")])

#confusion matrix
cm = table(test[,"BIC"], pred)


#importance
importance(rf)

#######Entropy

#train our random forest model
(rf <- randomForest(
  Entropy ~ Method + Sample.Size + M.Distance ,
  data=train, n.trees=501, keep.forest=FALSE
))


#we use our model to predict the testing data
pred = predict(rf, newdata=test[,c("Method", "Sample.Size",
                                   "M.Distance",  "E")])

#confusion matrix
cm = table(test[,"Entropy"], pred)

plot(cm)

#importance
importance(rf)


#plot random forest
options(repos='http://cran.rstudio.org')
have.packages <- installed.packages()
cran.packages <- c('devtools','plotrix','randomForest','tree')
to.install <- setdiff(cran.packages, have.packages[,1])
if(length(to.install)>0) install.packages(to.install)

library(devtools)
if(!('reprtree' %in% installed.packages())){
  install_github('araastat/reprtree')
}
for(p in c(cran.packages, 'reprtree')) eval(substitute(library(pkg), list(pkg=p)))


(rf <- randomForest(
  Entropy ~ Method + Sample.Size + M.Distance ,
  data=train, importance=TRUE, ntree=1, mtry = 2, do.trace=T
))


library(reprtree)
reprtree:::plot.getTree(rf)
