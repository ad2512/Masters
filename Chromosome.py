import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

# Importing Data
print "Importing Data"
data = np.loadtxt("bigdata.dat")
print "Data Imported"
nz = (data[:,0]==-1)
data = data[nz==0,:]
ny = (data[:,0]==25)
data = data[ny==0,:]
X_train = data[1:100000,1:31].astype('float32')
labels1 = data[1:100000,0].astype('int32')
y_train = np_utils.to_categorical(labels1)
X_test = data[100001:128990,1:31].astype('float32')
labels2 = data[100001:128990,0].astype('int32')
y_test = np_utils.to_categorical(labels2)

# Creating Model 
model = Sequential()
model.add(Dense(200, input_dim=30,init="uniform"))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(200,init="uniform"))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(25,init="uniform"))
model.add(Activation('softmax'))
#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer='RMSprop')
model.fit(X_train, y_train, batch_size=200, nb_epoch=20,verbose=1,show_accuracy=True,validation_data=(X_test,y_test))

# Testing model output
classes = model.predict_classes(data[:,1:31], batch_size=32)
proba = model.predict_proba(data[:,1:31], batch_size=32)
