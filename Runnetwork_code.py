#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import seed
import random
from csv import reader
from math import exp
from Wei_Shuang_network_code import*

#Set the parameters
n_inputs = 2
h = 10
n_outputs = 4
eta = 0.2
n_epoch = 200
n_folds = 2


seed(1)
#Test training backprop algorithm
training_filename = 'Wei_Shuang_dataset_training.txt'
training_dataset = load_csv(training_filename)
for i in range(len(training_dataset[0])-1):
	str_column_to_float(training_dataset, i)
str_column_to_int(training_dataset, len(training_dataset[0])-1)
network = neuralnetwork(n_inputs, h, n_outputs)
for layer in network:
	print layer
print 'The result of training network:'
train_mlp(network, training_dataset, eta, n_epoch, n_outputs)



# Test making predictions with the network
test_filename = 'Wei_Shuang_dataset_test.txt'
test_dataset = load_csv(test_filename)
for i in range(len(test_dataset[0])-1):
	str_column_to_float(test_dataset, i)
str_column_to_int(test_dataset, len(test_dataset[0])-1)
print 'The result of predicting the Class of the points:'
for inputs in test_dataset:
	num = 0
	prediction = classify_mlp(network, inputs)
	print 'Expected=%d, Predicted=%d' % (inputs[-1], prediction)


'''Wrong
Expected_Predicted = np.vstack((inputs[-1], prediction))
csvFile = open('prediction.txt','wb')
writer = csv.writer(csvFile, dialect='excel')
for i in range(len(inputs)):
	writer.writerow(Expected_Predicted)
csvFile.close()
'''


# Evaluate the accuracy of network
validation_filename = 'Wei_Shuang_dataset_validation.txt'
validation_dataset = load_csv(validation_filename)
for i in range(len(validation_dataset[0])-1):
	str_column_to_float(validation_dataset, i)
# convert class column to integers
str_column_to_int(validation_dataset, len(validation_dataset[0])-1)
# evaluate the error of network
print 'The result of evaluating the accuracy of network:'
scores = evaluate_mlp(validation_dataset, back_propagation, n_folds, eta, n_epoch, h)
print 'Scores: %s' % scores
print 'Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))
