import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\visha\OneDrive\Desktop\Data Science\Classroom\2.Apr 2025\21st Apr - SLR\21st- SLR\SLR - House price prediction\House_data.csv")

x = data[['sqft_living']]
y = data['price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.30,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title("sqft living vs price (test set")
plt.xlabel('sqft living')
plt.ylabel('price')
plt.show()

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title("sqft_living vs price (train set")
plt.xlabel("sqft living")
plt.ylabel("price")
plt.show()

slope = regressor.coef_
print(slope)

intercept = regressor.intercept_
print(intercept)

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test,y_test)
print(variance)

import pickle

filename = 'linear_regression_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor,file)
    
print("Model has been pickled and saved as linear_reg_model.pkl")
    