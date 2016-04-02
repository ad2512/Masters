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
from keras.optimizers import SGD, RMSprop, Adadelta
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
import cv2
from collections import Counter
import random

# Setting up the Data
A=662;
l = float(genfromtxt("/home/silo1/ad2512/FIDO/L" + str(1) + ".csv",delimiter=','))
l1 = float(genfromtxt("/home/silo1/ad2512/FIDO/L" + str(2) + ".csv",delimiter=','))
d = cv2.imread('/home/silo1/ad2512/FIDO/ZERONORM1.jpg')
d1 = cv2.imread('/home/silo1/ad2512/FIDO/ZERONORM2.jpg')
all_data=[d,d1]
labels=[l,l1]
for i in range(A-2):
	if((i+3)>A):
		break
	l = float(genfromtxt("/home/silo1/ad2512/FIDO/L" + str(i+3) + ".csv",delimiter=','))
	d = cv2.imread("/home/silo1/ad2512/FIDO/ZERONORM" + str(i+3) + ".jpg")
	all_data.append(d)
	labels.append(l)

s = np.shape(all_data)[1]
all_data = np.asarray(all_data)	
all_data = all_data.astype('float32')
all_data = all_data.reshape(A,3,s,s)
all_data /= np.max(np.abs(all_data),axis=0)
labels = np.asarray(labels)
labels = labels.astype('int')
nb_classes = np.size(np.unique(labels))
prop = 0.8;
train_labels=[]
train_data=[]
test_labels=[]
test_data=[]
c = Counter(labels)
def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest

for i in range(np.size(np.unique(labels))):
	a=[];
	b=[];
	for j in range(np.size(labels)):
		if(np.size(a)<prop*c[i]):
			if(labels[j]==i):
				a.extend([j])
		else:
			if(labels[j]==i):
				b.extend([j])
	train_labels.extend(labels[a])	
	train_data.extend(all_data[a])
	test_labels.extend(labels[b])
	test_data.extend(all_data[b])

a = range(np.size(train_labels))
b = scrambled(a)
train_data = [train_data[i] for i in b]
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
train_labels = [train_labels[i] for i in b]
labels = np_utils.to_categorical(labels)
c = Counter(train_labels)
print(c)
c = Counter(test_labels)
print(c)
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

# Building Model
# Building Model
model = Sequential()
model.add(Convolution2D(12,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('tanh'))
model.add(Convolution2D(12, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(24,3,3,border_mode='full'))
model.add(Activation('tanh'))
model.add(Convolution2D(24, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Convolution2D(48, 3, 3, border_mode='full'))
#model.add(Activation('tanh'))
#model.add(Convolution2D(48, 3, 3))
#model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
RMS = RMSprop(lr=0.0000000005, rho=0.7, epsilon=1e-08)
Ada = Adadelta(lr=0.001, rho=0.95, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=Ada)
model.fit(train_data, train_labels, batch_size=8, nb_epoch=200,verbose=1,show_accuracy=True,validation_data=(test_data, test_labels))

# Building Model
# Building Model
model = Sequential()
model.add(Convolution2D(12,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('tanh'))
model.add(Convolution2D(12, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(24,3,3,border_mode='full'))
model.add(Activation('tanh'))
model.add(Convolution2D(24, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Convolution2D(48, 3, 3, border_mode='full'))
#model.add(Activation('tanh'))
#model.add(Convolution2D(48, 3, 3))
#model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
RMS = RMSprop(lr=0.0000000005, rho=0.7, epsilon=1e-08)
Ada = Adadelta(lr=0.001, rho=0.95, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer = RMS)
model.fit(train_data, train_labels, batch_size=8, nb_epoch=200,verbose=1,show_accuracy=True,validation_data=(test_data, test_labels))
