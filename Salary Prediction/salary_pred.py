import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\visha\Downloads\Salary_Data.csv")

x = data.iloc[:,:-1].values

y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title("salary vs experiance (test set)")
plt.xlabel("years of experiance")
plt.ylabel("salary")
plt.show()

plt.scatter(x_train,y_train, color= 'red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experiance (train set")
plt.xlabel("Years of Experiance")
plt.ylabel("Salary")
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp = m_slope * 20 + c_intercept
print(pred_12yr_emp_exp)

pred_30yr_emp_exp = m_slope * 30 + c_intercept
print(pred_30yr_emp_exp)

# Training 
bias = regressor.score(x_train, y_train)
print(bias)

# Testing
variance = regressor.score(x_test, y_test)
print(variance)

# Stats for ML
data.mean()
data['Salary'].mean()

data.median()
data['Salary'].median()

data.mode()

data.var()
data['Salary'].var()

data.std()
data['Salary'].std()

# For calculating CV(Coefficient of Variation) we have to import scipy library
from scipy.stats import variation
variation(data.values)

variation(data['Salary'])


data.corr()

#SSR
y_mean = np.mean(y)
SSR = np.sum((y_pred - y_mean)**2)
print(SSR)

#SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(data.values)
SST = np.sum((data.values-mean_total)**2)
print(SST)

# R2
r_square = 1-SSR/SST
print(r_square)


import pickle
filename = 'linear_reg_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor,file)

print("Model has been pickled and saved as linear_reg_model.pkl")

import os
os.getcwd()
