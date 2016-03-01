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
from matplotlib.pyplot import imshow

# Setting up the data
(train, labels_1), (test, labels_2) = mnist.load_data()
train = train.reshape(60000,1,28,28)
train = train.astype('float32')
test = test.reshape(10000,1,28,28)
test = test.astype('float32')
labels_1 = np_utils.to_categorical(labels_1)
labels_2a = np_utils.to_categorical(labels_2)

# Building Model - Model 1
model1 = Sequential()
model1.add(Convolution2D(16,5,5,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Convolution2D(32,5,5,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Flatten())
model1.add(Dropout(0.5))
model1.add(Dense(1000))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1000))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(10))
model1.add(Activation('softmax'))
model1.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model1.fit(train, labels_1, batch_size=300, nb_epoch=5,verbose=1,show_accuracy=True,validation_data=(test, labels_2a))
classes1 = model1.predict_classes(test, batch_size=300)

# Building Model - Model 2
model2 = Sequential()
model2.add(Convolution2D(16,3,3,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Convolution2D(32,3,3,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Convolution2D(64,3,3,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(1000))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10))
model2.add(Activation('softmax'))
model2.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model2.fit(train, labels_1, batch_size=300, nb_epoch=5,verbose=1,show_accuracy=True,validation_data=(test, labels_2a))
classes2 = model2.predict_classes(test, batch_size=300)

# Building Model - Model 3
model3 = Sequential()
model3.add(Convolution2D(16,3,3,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Convolution2D(32,3,3,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Flatten())
model3.add(Dropout(0.5))
model3.add(Dense(1000))
model3.add(Activation('relu'))
model3.add(Dropout(0.5))
model3.add(Dense(1000))
model3.add(Activation('relu'))
model3.add(Dropout(0.5))
model3.add(Dense(10))
model3.add(Activation('softmax'))
model3.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model3.fit(train, labels_1, batch_size=300, nb_epoch=5,verbose=1,show_accuracy=True,validation_data=(test, labels_2a))
classes3 = model3.predict_classes(test, batch_size=300)

# Building Model - Model 4
model4 = Sequential()
model4.add(Convolution2D(16,5,5,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Convolution2D(50,5,5,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(3,3)))
model4.add(Flatten())
model4.add(Dropout(0.5))
model4.add(Dense(1000))
model4.add(Activation('relu'))
model4.add(Dropout(0.5))
model4.add(Dense(10))
model4.add(Activation('softmax'))
model4.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model4.fit(train, labels_1, batch_size=300, nb_epoch=5,verbose=1,show_accuracy=True,validation_data=(test, labels_2a))
classes4 = model4.predict_classes(test, batch_size=300)


# Building Model - Model 5
model5 = Sequential()
model5.add(Convolution2D(16,5,5,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model5.add(Activation('relu'))
model5.add(MaxPooling2D(pool_size=(2,2)))
model5.add(Convolution2D(32,5,5,init='uniform',border_mode='valid',input_shape=(1,28,28)))
model5.add(Activation('relu'))
model5.add(MaxPooling2D(pool_size=(2,2)))
model5.add(Flatten())
model5.add(Dropout(0.5))
model5.add(Dense(1000))
model5.add(Activation('relu'))
model5.add(Dropout(0.5))
model5.add(Dense(1000))
model5.add(Activation('relu'))
model5.add(Dropout(0.5))
model5.add(Dense(10))
model5.add(Activation('softmax'))
model5.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model5.fit(train, labels_1, batch_size=300, nb_epoch=5,verbose=1,show_accuracy=True,validation_data=(test, labels_2a))
classes5 = model5.predict_classes(test, batch_size=300)

# Calculating modes
classes_final = []
for i in range(10000):
	from collections import Counter
	c = Counter([classes1[i],classes2[i],classes3[i],classes4[i],classes5[i]])
	classes_final[i] = max(c.items(), key=lambda x:x[1])[0]
	
count = 0
for i in range(10000):
	if(classes_final[i]!=labels_2[i]):
		count = count + 1/10000
		
print(count)