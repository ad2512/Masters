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
import cv2

# Setting up the Data
A=539;
l = float(genfromtxt("/home/silo1/ad2512/Histo_6/L" + str(1) + ".csv",delimiter=','))
l1 = float(genfromtxt("/home/silo1/ad2512/Histo_6/L" + str(2) + ".csv",delimiter=','))
d = cv2.imread('/home/silo1/ad2512/Histo_6/NORM1.jpg')
d1 = cv2.imread('/home/silo1/ad2512/Histo_6/NORM2.jpg')
all_data=[d,d1]
labels=[l,l1]
for i in range(A-2):
	if((i+3)>A):
		break
	l = float(genfromtxt("/home/silo1/ad2512/Histo_6/L" + str(i+3) + ".csv",delimiter=','))
	d = cv2.imread("/home/silo1/ad2512/Histo_6/NORM" + str(i+3) + ".jpg")
	all_data.append(d)
	labels.append(l)

s = np.shape(all_data)[1]
all_data = np.asarray(all_data)	
all_data = all_data.astype('float32')
all_data = all_data.reshape(A,3,s,s)
labels = np.asarray(labels)
labels = labels.astype('int')
labels = np_utils.to_categorical(labels)
print ("0 = %s",np.shape(all_data)[0])
print ("0 = %s",np.shape(all_data)[1])
print ("0 = %s",np.shape(all_data)[2])
print ("0 = %s",np.shape(all_data)[3])


print ("s = %s",s)


# Building Model
# Building Model
model = Sequential()
model.add(Convolution2D(32,3,3,init='uniform',border_mode='full',input_shape=(3,s,s)))
model.add(Activation('relu'))
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, border_mode='full'))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
#model.add(Dense(500))
#model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(6))
model.add(Activation('softmax'))

sgd = SGD(lr=0.0000003, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)
model.fit(all_data, labels, batch_size=10, nb_epoch=50,verbose=1,show_accuracy=True,validation_data=(all_data[300:539], labels[300:539]))
