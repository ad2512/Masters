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
from numpy import append, array, int8, uint8, zeros

# Setting up the data
(train, labels_1), (test, labels_2) = mnist.load_data()
train = train.reshape(60000,1,28,28)
train = train.astype('float32')
test = test.reshape(10000,1,28,28)
test = test.astype('float32')
labels_1 = np_utils.to_categorical(labels_1)
labels_2 = np_utils.to_categorical(labels_2)


# Building Model - Note that model.add(Activation('relu')) doesn't work when it should. Problem with dimensions
# model = Sequential()
# model.add(Convolution2D(16,2,2,init='uniform',border_mode='valid',input_shape=(1,28,28)))
# model.add(Activation('relu'))
# model.add(Convolution2D(16,2,2))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(16,2,2))
# model.add(Activation('relu'))
# model.add(Convolution2D(16,2,2))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dropout(0.3))
# model.add(Dense(10))
# model.add(Activation('softmax'))

model = Sequential()
model.add(Convolution2D(16,2,2,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.003, decay=0.0002,nestorov=True)
#rms = RMSprop(lr=0.001, rho=0.95, epsilon=1e-15)
model.compile(loss='categorical_crossentropy', optimizer="RMSprop")

model.fit(train, labels_1, batch_size=300, nb_epoch=100,verbose=1,show_accuracy=True,validation_data=(test, labels_2))


