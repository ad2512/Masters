from __future__ import absolute_import
from __future__ import print_function
import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.datasets import mnist
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
import pylab as pl
import matplotlib.cm as cm
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros,genfromtxt
from matplotlib.pyplot import imshow
from sklearn.cross_validation import train_test_split
from random import randint

# Setting up the Data
data = genfromtxt("N" + str(1) + ".csv",delimiter=',')
s = np.shape(data)[0]
a = data[0:s,0:s]
b = data[0:s,s:(2*s)]
c = data[0:s,(2*s):(3*s)]
d = np.dstack((a,b,c))
data = genfromtxt("N" + str(2) + ".csv",delimiter=',')
a = data[0:s,0:s]
b = data[0:s,s:(2*s)]
c = data[0:s,(2*s):(3*s)]
d1 = np.dstack((a,b,c))
all_data=[d,d1]
labels=genfromtxt("Labels.csv",delimiter=',')
for i in range(np.size(labels)-2):
	print(i+3)
	data = genfromtxt("N" + str(i+3) + ".csv",delimiter=',')
	a = data[0:s,0:s]
	b = data[0:s,s:(2*s)]
	c = data[0:s,(2*s):(3*s)]
	d = np.dstack((a,b,c))
	all_data.append(d)
	
	
train, test, labels_1, labels_2 = train_test_split(all_data,labels,test_size=0.4)
train = train.reshape(np.shape(train)[0],3,s,s)
train = train.astype('float32')
test_r=test
test = test.reshape(np.shape(test)[0],3,s,s)
test = test.astype('float32')
labels_1 = np_utils.to_categorical(labels_1)
labels_2a = np_utils.to_categorical(labels_2)

# Building Model
model = Sequential()
model.add(Convolution2D(8,5,5,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(32, 5, 5, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(64, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model.fit(train, labels_1, batch_size=20, nb_epoch=50,verbose=1,show_accuracy=True,validation_data=(test, labels_2a))



model = Sequential()
model.add(Convolution2D(8,5,5,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(32, 5, 5, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(64, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model.fit(train, labels_1, batch_size=20, nb_epoch=50,verbose=1,show_accuracy=True,validation_data=(test, labels_2a))