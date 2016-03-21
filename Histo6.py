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
from numpy import append, array, int8, uint8, zeros,genfromtxt, matrix
from matplotlib.pyplot import imshow
from sklearn.cross_validation import train_test_split
from random import randint

# Setting up the Data
A=539;
data = genfromtxt("/home/silo1/ad2512/Histo_6/N" + str(1) + ".csv",delimiter=',')
l = float(genfromtxt("/home/silo1/ad2512/Histo_6/L" + str(1) + ".csv",delimiter=','))
s = np.shape(data)[0]
a = data[0:s,0:s]
b = data[0:s,s:(2*s)]
c = data[0:s,(2*s):(3*s)]
d = np.dstack((a,b,c))
data = genfromtxt("/home/silo1/ad2512/Histo_6/N" + str(2) + ".csv",delimiter=',')
l1 = float(genfromtxt("/home/silo1/ad2512/Histo_6/L" + str(2) + ".csv",delimiter=','))
a = data[0:s,0:s]
b = data[0:s,s:(2*s)]
c = data[0:s,(2*s):(3*s)]
d1 = np.dstack((a,b,c))
all_data=[d,d1]
labels=[l,l1]
for i in range(A-2):
	print(i+3)
	if((i+3)>A):
		break
	data = genfromtxt("/home/silo1/ad2512/Histo_6/N" + str(i+3) + ".csv",delimiter=',')
	l = float(genfromtxt("/home/silo1/ad2512/Histo_6/L" + str(i+3) + ".csv",delimiter=','))
	a = data[0:s,0:s]
	b = data[0:s,s:(2*s)]
	c = data[0:s,(2*s):(3*s)]
	d = np.dstack((a,b,c))
	all_data.append(d)
	labels.append(l)
	
all_data = np.asarray(all_data)	
all_data = all_data.astype('float32')
all_data = all_data.reshape(A,3,s,s)
labels = np.asarray(all_data)
labels = labels.astype('int')
labels = np_utils.to_categorical(labels)

# Building Model
model = Sequential()
model.add(Convolution2D(8,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#model.add(Convolution2D(32, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(64, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model.fit(all_data[0:400], labels[0:400], batch_size=5, nb_epoch=20,verbose=1,show_accuracy=True,validation_data=(all_data[400:539], labels[400:539]))



# Building Model
model = Sequential()
model.add(Convolution2D(16,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#model.add(Convolution2D(32, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(64, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model.fit(all_data[0:400], labels[0:400], batch_size=5, nb_epoch=20,verbose=1,show_accuracy=True,validation_data=(all_data[400:539], labels[400:539]))


# Building Model
model = Sequential()
model.add(Convolution2D(16,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(64, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model.fit(all_data[0:400], labels[0:400], batch_size=5, nb_epoch=20,verbose=1,show_accuracy=True,validation_data=(all_data[400:539], labels[400:539]))



# Building Model
model = Sequential()
model.add(Convolution2D(16,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#model.add(Convolution2D(32, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(64, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model.fit(all_data[0:400], labels[0:400], batch_size=5, nb_epoch=20,verbose=1,show_accuracy=True,validation_data=(all_data[400:539], labels[400:539]))



# Building Model
model = Sequential()
model.add(Convolution2D(16,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#model.add(Convolution2D(32, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(64, 5, 5, border_mode='full'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop")
model.fit(all_data[0:400], labels[0:400], batch_size=5, nb_epoch=20,verbose=1,show_accuracy=True,validation_data=(all_data[400:539], labels[400:539]))