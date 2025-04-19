# Step 1: Import the Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Step 2: Import the Dataset & Divide the Dataset into Dependent and Independent 

dataset = pd.read_csv(r"C:\Users\visha\Downloads\data (2).csv") 

x = dataset.iloc[:,:-1].values

y = dataset.iloc[:,3].values

# sklearn --> scikit learn a popular python library for ML

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

imp = imputer.fit(x[:,1:3]) # Analyze the data in the 2 columns

x[:,1:3] = imp.transform(x[:,1:3]) # calculates mean and replaces all Nan values

# Encode Categorical Data & Create a Dummy Variable

from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0])

x[:,0] = labelencoder_x.transform(x[:,0])

# --------------------------------------------------

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

# -------------------------------------------------
# Split the Dataset into Trainig & Testing set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x, y, train_size=0.7,random_state=0)






























