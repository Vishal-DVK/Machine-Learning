import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Libraries 
data = pd.read_csv(r"C:\Users\visha\Downloads\12th\30th, 31st\Social_Network_Ads.csv")

# Split the data
x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

# Train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# Feature Scaling 
from sklearn.preprocessing import Normalizer
nz = Normalizer()
x_train = nz.fit_transform(x_train)
x_test = nz.fit_transform(x_test)

# Train the model 
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)

bias = nb.score(x_train,y_train)
print(bias)

variance = nb.score(x_test,y_test)
print(variance)


# Visualizing the training set results
from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2 = np.meshgrid(
    np.arange(start = x_set[:,0].min() -1, stop = x_set[:,0].max() +1, step = 0.01),
    np.arange(start = x_set[:,1].min() -1, stop = x_set[:,1].max() +1, step = 0.01)
    )

plt.contourf(
    x1,x2,
    nb.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75,
    cmap = ListedColormap(('red','green'))
    )

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1], c = ListedColormap(('red','green'))(i), label = j)

plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
   

# Visualization for test set results
from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(
    np.arange(start = x_set[:,0].min() -1, stop = x_set[:,0].max() +1, step = 0.01),
    np.arange(start = x_set[:,1].min() -1, stop = x_set[:,1].max() +1, step = 0.01)
    )

plt.contourf(
    x1,x2,
    nb.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75,
    cmap = ListedColormap(('red','green'))
    )

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1], c = ListedColormap(('red','green'))(i), label = j)

plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()          