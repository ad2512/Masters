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


data = genfromtxt("N" + str(i+1) + ".csv",delimiter=',')
s = np.shape(data)[0]
all_data=[]
for i in range(696):
	data = genfromtxt("N" + str(i+1) + ".csv",delimiter=',')
	a = data[0:s,0:s]
	b = data[0:s,s:(2*s)]
	c = data[0:s,(2*s):(3*s)]
	d = np.dstack((a,b,c))
	all_data = all_data.append(d)
	
train, test, labels_1, labels_2 = train_test_split(all_data,labels,test_size=0.2)
train = train.reshape(np.shape(train)[0],3,s,s)
train = train.astype('float32')
test_r=test
test = test.reshape(np.shape(test)[0],3,s,s)
test = test.astype('float32')
labels_1 = np_utils.to_categorical(labels_1)
labels_2a = np_utils.to_categorical(labels_2)
