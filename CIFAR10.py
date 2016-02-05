import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

# Setting up the data
(train, labels_1), (test, labels_2) = cifar10.load_data()
train = train.astype('float32')
test = test.astype('float32')
labels_1 = np_utils.to_categorical(labels_1)
labels_2 = np_utils.to_categorical(labels_2)


# Building Model
model = Sequential()
model.add(Convolution2D(30,2,2,init='uniform',border_mode='full',input_shape=(3,32,32)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('tanh'))
model.add(Convolution2D(30, 2, 2))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, border_mode='full'))
model.add(Activation('tanh'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('softmax'))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(train, labels_1, batch_size=300, nb_epoch=30,verbose=1,show_accuracy=True,validation_data=(test, labels_2))




