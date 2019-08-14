# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:38:46 2019

@author: reuve
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

#calcalulate  mean and std per lable
def mean_and_std(x, y, label):
    x_classified = x[y==label]
#   print('x_classified:\n',x_classified)
    mean = np.mean(x_classified, axis=0)
    std = np.std(x_classified, axis=0)
    return (mean, std)  
  
#calc mean and std for train and test sets   
def calc_mean_std(x, y):
    mean_label_0, std_label_0 = mean_and_std(x, y , 0)
    mean_label_1, std_label_1 = mean_and_std(x, y, 1)
    return (mean_label_0, std_label_0, mean_label_1, std_label_1)   

#test
#mean0,std0,mean1,std1 = calc_mean_std(x_train,y_train)

#3
# based on equation of Gaussian probability
def gussian_prob(x, mean, std):
    # calc probabilities of a row
    exponent = np.exp( -(x-mean)**2/(2*std**2) )
    prob_array = ( 1/( np.sqrt(2*math.pi)*std) ) * exponent 
#    print('prob_array\n', prob_array.shape)
    # reshape to a colunm (size,1)
    prob_array = np.reshape(prob_array, (prob_array.shape[0],1))
    #print('prob_array\n', prob_array.shape)
    # calc the product 
    prob = np.prod(prob_array, axis=0)
    #print('prob_\n', prob)       
    return (prob)

#4
# calc predictions   
def prediction(x_values, mean0, std0, mean1, std1):    
    prob0 = []
    prob1 = []
    for row in x_values:
        prob0_row = gussian_prob(row, mean0, std0)
        prob1_row = gussian_prob(row, mean1, std1)
        prob0.append(prob0_row)
        prob1.append(prob1_row)
    probabilities = np.column_stack( (prob0, prob1) )
 #   print('probabilities\n',probabilities)
    y_predicted = np.argmax(probabilities, axis=1)
#    print('y_predicted:\n',y_predicted)
    return (y_predicted)

#5
def evaluate_accuracy(x_test, y_test, mean0, std0, mean1, std1):
    y_calculated = prediction(x_test, mean0, std0, mean1, std1)    
    diff = y_calculated - y_test
    wrong_points = np.sum(np.sqrt(diff**2) )
    length = y_calculated.shape[0]
    accuracy = 1-wrong_points/length
    return (accuracy)  
 
#6    
class naive_bayes_class():

    def __init__(self):
        self.alg_name = 'NaiveBayes'
        self.mean0 = ''
        self.std0 = ''
        self.mean1 = ''
        self.std1 = ''
            
    def fit(self, x_train, y_train):
        mean0,std0,mean1,std1 = calc_mean_std(x_train, y_train)
        self.mean0 = mean0
        self.mean1 = mean1
        self.std0 = std0
        self.std1 = std1
        
    def predict(self, x_test):
        y_calculated = prediction(x_test, self.mean0, self.std0, self.mean1, self.std1)   
        return (y_calculated)
    
    def evaluate(self,x_test,y_test):
        accuracy = evaluate_accuracy(x_test, y_test, self.mean0, self.std0, self.mean1, self.std1)
        return (accuracy) 
 
#main
file_name = 'C:/ML bootcamp/ML/Naive Bayes/pima-indians-diabetes.csv'
data = pd.read_csv(file_name).values
X = data[:,:-1]
Y = data[:,-1]
#x_train,x_test,y_train,y_test = make_train_test(X, Y, 0.8)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)    
 
#2
#Calc mean and std variance for each X column
mean_x = np.mean(X, axis=0)
std_x = np.std(X, axis=0)
print('mean_x:\n',mean_x)
print('std_x:\n',std_x)

# create the class
NBC = naive_bayes_class()
# train data
NBC.fit(x_train,y_train)
# calc accuracy
accuracy = NBC.evaluate(x_test, y_test)
print('accuracy: ', accuracy)    
