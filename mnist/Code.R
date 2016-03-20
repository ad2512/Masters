# Setting up the environment
setwd("C:/Users/Adam/Documents/R/")
load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train1 <<- load_image_file('mnist/train-images.idx3-ubyte')
  test1 <<- load_image_file('mnist/t10k-images.idx3-ubyte')
  
  train1$y <<- load_label_file('mnist/train-labels.idx1-ubyte')
  test1$y <<- load_label_file('mnist/t10k-labels.idx1-ubyte')  
}
load_mnist()
library("randomForest", lib.loc="~/R/win-library/3.1")
labels <- as.factor(train1$y)
train.var <- train1$x
rf.bench4 <- randomForest(train.var,labels,ntree=5000,keep.inbag=TRUE)
show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}


# Random Forests

library("randomForest", lib.loc="~/R/win-library/3.1")
labels <- as.factor(train1$y)
train.var <- train1$x
# rf.bench1 <- randomForest(train.var,labels,ntree=100) - train = 3.46% - test = 3.09%
#rf.bench2 <- randomForest(train.var,labels,ntree=100,importance=TRUE) - train = 3.48% - test = 2.92%
#rf.bench3 <- randomForest(train.var,labels,ntree=100,corr.bias=TRUE) - train = 3.45% - test = 3.10%
#rf.bench4 <- randomForest(train.var,labels,ntree=100,keep.inbag=TRUE) - train = 3.40% - test = 2.89%
a <- predict(rf.bench4,test1$x)
count<-0
for(i in 1:10000){
     if(a[i]!=test1$y[i]){
         count <- count+1/10000;
     }
}
count

# H2O

h2otrain <- as.h2o(cbind(train1$y,train1$x))
