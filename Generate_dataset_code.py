#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv



plt.figure('Figure',figsize=(15,10))
#plt.figure('Figure1')
n = 500
#Class 1
X1 = np.random.rand(n)*2+1
Y1 = np.random.rand(n)*4-5
C1 = (np.array([X1,Y1])).T
#C1 = C1[np.newaxis,:]
#plt.scatter(X1,Y1,c='b', s=25, alpha=.5)

#Class 2
X2 = np.random.rand(n)*3+2
Y2 = np.random.rand(n)*3+1
C2 = (np.array([X2,Y2])).T
#C2 = C2[np.newaxis,:]
#plt.scatter(X2,Y2, c='r', s=25, alpha=.5, marker = 'x')


#Rotate the figure1 in figure2
#plt.figure('Figure2')
#rotate C1
for i in range(0,499):
	X1ii = X1[i]
	Y1ii = Y1[i]
	P2 = [X1ii,Y1ii]
	#P = np.array(C1)
	theta = (math.pi)/(12)
	R = np.array([[math.cos(theta),-math.sin(theta)], [math.sin(theta), math.cos(theta)]])
	C1_dot = np.dot(P2, R)
	C1_dot_i0 = C1_dot[0]
	C1_dot_i1 = C1_dot[1]
	plt.scatter(C1_dot_i0,C1_dot_i1,c='b', s=50, alpha=.5)
#rotate C2
for i in range(0,499):
	X2ii = X2[i]
	Y2ii = Y2[i]
	P2 = [X2ii,Y2ii]
	theta = (math.pi)/(12)
	R = np.array([[math.cos(theta),-math.sin(theta)], [math.sin(theta), math.cos(theta)]])
	C2_dot = np.dot(P2, R)
	C2_dot_i0 = C2_dot[0]
	C2_dot_i1 = C2_dot[1]
	plt.scatter(C2_dot_i0,C2_dot_i1,c='r', s=50, alpha=.5, marker = 'x')


#generate class3 and class4
#plt.figure('Figure3')
#Class 3
mean3 = [-2, -3]
cov3 = [[0.5, 0], [0, 3]]
X3,Y3 = np.random.multivariate_normal(mean3, cov3, (500)).T
#Y3 = np.random.multivariate_normal(mean3, cov3, (250)).T
C3 = (np.array([X3,Y3])).T
#C3 = np.hstack((X3,Y3))
#C3 = np.array([x[0] for x in C3])
plt.scatter(X3,Y3 ,c='g', s=50, alpha=.5)

#Class 4
mean4 = [-4, -1]
cov4 = [[3, 0.5], [0.5, 0.5]]
X4,Y4 = np.random.multivariate_normal(mean4, cov4, (500)).T
C4 = (np.array([X4, Y4])).T
#C4 = np.hstack((X4,Y4))
#C4 = np.array(C4)
plt.scatter(X4,Y4, c='cyan', s=50, alpha=.5)


plt.xlim(-10,7)
plt.ylim(-10,7)

plt.show()


#Add to each point vector an element with the class, from 1 to 4
arr1 = np.linspace(1, 1, 500)
C1_new = np.column_stack((C1,arr1))
#print arr1.shape
#print C1.shape

arr2 = np.linspace(2, 2, 500)
C2_new = np.column_stack((C2,arr2))

arr3 = np.linspace(3, 3, 500)
C3_new = np.column_stack((C3,arr3))

arr4 = np.linspace(4, 4, 500)
C4_new = np.column_stack((C4,arr4))


'''
print C1_new.shape
print C2_new.shape
print C3_new.shape
print C4_new.shape
'''

#Aggregate all the points into a single matrix of size 2000*3
dataset = np.vstack((C1_new,C2_new,C3_new,C4_new))
#print dataset.shape
np.random.shuffle(dataset)

training, validation, test = np.split(dataset,[1000,1500])

'''
print training.shape
print validation.shape
print test.shape
'''

def generate_dataset():
	return training, validation, test 

#save dataset_training
csvFile = open('Wei_Shuang_dataset_training_new.txt','wb')
writer = csv.writer(csvFile,dialect='excel')
for i in range(len(training)):
	writer.writerow(training[i])
csvFile.close()

#save dataset_test
csvFile = open('Wei_Shuang_dataset_test_new.txt','wb')
writer = csv.writer(csvFile,dialect='excel')
for i in range(len(test)):
	writer.writerow(test[i])
csvFile.close()

#save dataset_validation
csvFile = open('Wei_Shuang_dataset_validation_new.txt','wb')
writer = csv.writer(csvFile,dialect='excel')
for i in range(len(validation)):
	writer.writerow(validation[i])
csvFile.close()

print 'Save dataset successfully.'
