this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
print(getwd())
dat <- read.csv(file="../../data/Dtrain.csv", header=TRUE)
