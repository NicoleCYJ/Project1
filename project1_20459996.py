# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:01:04 2017

@author: Nicole
"""
import os
import pandas as pd
import numpy
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data set as matrix
train = pd.read_csv('./traindata.csv',sep = ',',encoding = 'utf-8', header=None)
train = train.as_matrix()
label = pd.read_csv('./trainlabel.csv',sep = ',',encoding = 'utf-8', header=None)
label = label.as_matrix()
test = pd.read_csv('./testdata.csv',sep = ',',encoding = 'utf-8', header=None)
test = test.as_matrix()

# Pre-process data
train[:, 54:57] = preprocessing.scale(train[:, 54:57], axis = 0)
test[:, 54:57] = preprocessing.scale(test[:, 54:57], axis = 0)

# Split initial train data as new train data and validation data
train, validation, label, validation_label = train_test_split(train, label[:, 0], random_state=0)

# Use AdaBoost based on Decision Tree to build the model1
model1 = AdaBoostClassifier(n_estimators = 90)
model1.fit(train, label)
accuracy = model1.score(validation, validation_label)
print("AdaBoost Classifier: ", accuracy)

# Use Random Forest Classifier to build the model2
model2 = RandomForestClassifier(n_estimators = 15)
model2.fit(train, label)
accuracy = model2.score(validation, validation_label)
print("RandomForest Classifier: ", accuracy)

# Use Multi-layer Perceptron Classifier to build model3
model3 = MLPClassifier(activation = 'tanh')
model3.fit(train, label)
accuracy = model3.score(validation, validation_label)
print("MLPClassifier Classifier: ", accuracy)

# Ensemble model1, model2 and model3 to predict the validation data 
# and compute the ensembled accuracy 
right = 0
for i in range(len(validation_label)):
    y1 = model1.predict(validation[i, :].reshape(1, -1))
    y2 = model2.predict(validation[i, :].reshape(1, -1))
    y3 = model3.predict(validation[i, :].reshape(1, -1))
    if y1+y2+y3 < 2:
        y = 0
    else:
        y = 1
    if y == validation_label[i]:
        right += 1
accuracy = right / len(validation_label)
print("Ensemble previous models: ", accuracy)

# Ensemble model1, model2 and model3 to predict the test data
[a, b] = numpy.shape(test)
result = []
for j in range(a):
    y1 = model1.predict(test[j, :].reshape(1, -1))
    y2 = model2.predict(test[j, :].reshape(1, -1))
    y3 = model3.predict(test[j, :].reshape(1, -1))
    if y1+y2+y3 < 2:
        result.append(0)
    else:
        result.append(1)

# Write the prediction of the test data into a .csv file
with open('./project1_20459996.csv','w')as f:
    writer=csv.writer(f,lineterminator='\n')
    for val in result:
        writer.writerow([val])
