# This file deals with reading the data from local csv files
# as well as formatting them in the desired way.

import pandas as pd 
import numpy as np
import os

TRAINING_DATA = os.path.join(os.path.dirname(__file__), 'data/train.csv')
TESTING_DATA = os.path.join(os.path.dirname(__file__), 'data/test.csv')

TRAINING_DATA_200k = os.path.join(os.path.dirname(__file__), 'data/train_20k.csv')

# Read the default set of data
def get_default_data():
	return get_data(TRAINING_DATA, TESTING_DATA)

def get_200k_data():
	return get_training_data(TRAINING_DATA_200k)

# Join two set of data
# Notice that the return type is a two tuple
def join_data(data1, data2):
	train_X = np.vstack((data1[0], data2[0]))
	train_y = np.hstack((data1[1], data2[1]))
	return (train_X, train_y)

def get_data(train, test):
	train_X, train_y = get_training_data(train)
	test_X = get_testing_data(test)
	
	return (train_X, train_y, test_X)

def get_training_data(path):
	train_raw = pd.read_csv(path)
	train_X = train_raw.ix[:,1:].values.astype("int32")
	train_y = train_raw.ix[:,0].values.astype("int32")
	print "Loaded training data"
	return (train_X, train_y)

def get_testing_data(path):
	test_raw = pd.read_csv(path)
	test_X = test_raw.ix[:,:].values.astype("int32")
	print "Loaded testing data"
	return (test_X)