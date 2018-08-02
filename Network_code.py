#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import random
from random import seed
from random import randrange
from csv import reader
from math import exp
import csv
#from Generate_dataset_code import *

#According to the requirement of the coursework, I use following symbol to represent each meaning in this network code.
#x: input a point(inputs n), w: the vector of w, h: the number of neurons in the hidden layer, n_outputs: the number of neurons in the output layer(outputs n), eta: a learning rate


#Initialize a network
def neuralnetwork(n_inputs, h, n_outputs):
	network = list()
	hidden_layer = [{'w':[random.uniform(-0.5,0.5) for i in range(n_inputs + 1)]}
	 for j in range(h)]
	network.append(hidden_layer)
	output_layer = [{'w':[random.uniform(-0.5,0.5) for i in range(h + 1)]}
	 for j in range(n_outputs)]
	network.append(output_layer)
	return network

'''Wrong record
def neuralnetwork(n_inputs, h, n_outputs):
	network = list()
	for i in range(h):
		for j in range(n_inputs + 1):
			hidden_layer[j] = random.uniform(-0.5,0.5)
		hidden_layer = [{'w':hidden_layer[j]}]
	network.append(hidden_layer)
	for i in range(n_outputs):
		for j in range(h + 1):
			output_layer[j] = random.uniform(-0.5,0.5)
		output_layer = [{'w':output_layer[j]}]
	network.append(output_layer)
	return network
'''

#Calculate neuron activation for an input
def activate(w, x):
	activation = w[-1]
	for i in range(len(w) - 1):
		activation = activation + w[i] * x[i]
	return activation

#Set up neuron activation function
def activation_function(activation):
	return 1.0 / (1.0 + exp(-activation))

#Calculate the derivative of an neuron output
def activation_function_derivative(output):
	return output * (1.0 - output)

#Forward proagate input to a network output
def forward_propagate(network, inputs):
	x = inputs
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['w'], x)
			neuron['output'] = activation_function(activation)
			new_inputs.append(neuron['output'])
		x = new_inputs
	return x


#Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error = error + (neuron['w'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * activation_function_derivative(neuron['output'])

#Update network weights with error
def update_w(network, inputs, eta):
	for i in range(len(network)):
		x = inputs[:-1]
		if i != 0:
			x = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(x)):
				neuron['w'][j] = neuron['w'][j] + eta * neuron['delta'] * x[j]
			neuron['w'][-1] = neuron['w'][-1] + eta *neuron['delta']

#Update network weights with error
def train_mlp(network, train, eta, n_epoch, n_outputs):
	list_epoch = []
	list_error = []
	list_epochanderror = []
	for epoch in range(n_epoch):
		sum_error = 0
		for inputs in train:
			outputs = forward_propagate(network, inputs)
			expected = [0 for i in range(n_outputs)]
			expected[inputs[-1]] = 1
			sum_error = sum_error + 0.5 * sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_w(network, inputs, eta)
		print '>epoch =%d, eta =%.3f, error =%.3f' % (epoch, eta, sum_error)
		#Save the epoch and error into csv
		list_epoch.append(epoch)
		list_error.append(sum_error)
		list_epochanderror = np.stack((list_epoch, list_error)).T
		#np.savetxt('epochanderror.csv', list_epochanderror)    txt isn't good at analysis
		csvFile = open('epochanderror.csv','wb')
		writer = csv.writer(csvFile,dialect='excel')
		for i in range(len(list_epochanderror)):
			writer.writerow(list_epochanderror[i])
		csvFile.close()

#Make prediction with network
def classify_mlp(network, inputs):
	outputs = forward_propagate(network, inputs)
	return outputs.index(max(outputs))

#Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, eta, n_epoch, h):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([inputs[-1] for inputs in train]))
	network = neuralnetwork(n_inputs, h, n_outputs)
	train_mlp(network, train, eta, n_epoch, n_outputs)
	predictions = list()
	for inputs in test:
		prediction = classify_mlp(network, inputs)
		predictions.append(prediction)
	return(predictions)

#Split a dataset into k folds
def kfold_cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

#Calculate accuracy
def accuracy_function(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct = correct + 1
	return correct / float(len(actual)) * 100.0

#Evaluate using cross validation split
def evaluate_mlp(dataset, back_propagation_algorithm, n_folds, *args):
	folds = kfold_cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for inputs in fold:
			inputs_copy = list(inputs)
			test_set.append(inputs_copy)
			inputs_copy[-1] = None
		predicted = back_propagation_algorithm(train_set, test_set, *args)
		actual = [inputs[-1] for inputs in fold]
		accuracy = accuracy_function(actual, predicted)
		scores.append(accuracy)
	return scores


#Other function for load dataset

#Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for inputs in csv_reader:
			if not inputs:
				continue
			dataset.append(inputs)
	return dataset

#Convert string column to float
def str_column_to_float(dataset, column):
	for inputs in dataset:
		inputs[column] = float(inputs[column].strip())

#Convert string column to int
def str_column_to_int(dataset, column):
	class_values = [inputs[column] for inputs in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for inputs in dataset:
		inputs[column] = lookup[inputs[column]]
	return lookup
