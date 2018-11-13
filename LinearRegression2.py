# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:15:24 2018

@author: Admin
"""

# Python code to illustrate 
# regression using data set 
#import matplotlib 
#matplotlib.use('GTKAgg') 


import matplotlib.pyplot as plt 

from sklearn import preprocessing,linear_model , svm

import pandas as pd 
import numpy as np
import math

# Load CSV and columns 
df = pd.read_csv("HouseData.csv") 

X = df['plotsize'] 
Y = df['price'] 




#Transposing the data in the X and Y
X=X.values.reshape(len(X),1) 
Y=Y.values.reshape(len(Y),1) 

# Split the data into training/testing sets 
X_train = X[:-250] 
X_test = X[-250:] 

# Split the targets into training/testing sets 
Y_train = Y[:-250] 
Y_test = Y[-250:] 


# Create linear regression object 
regr = linear_model.LinearRegression() 


# Train the model using the training sets 
regr.fit(X_train, Y_train) 

# Plot outputs 
plt.scatter(X_test, Y_test, color='black') 
plt.title('Linear Regression Model') 
plt.xlabel('Size') 
plt.ylabel('Price') 
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3) 
plt.show() 


plot_size = 6000
print("Plot size:", plot_size, "\nHouse Price:",str(regr.predict(plot_size)))


