import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\visha\Downloads\logit classification.csv")

x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(x_train,y_train)
print(bias)

variance = classifier.score(x_test,y_test)
print(variance)

from sklearn.metrics import roc_curve, roc_auc_score
y_prob = classifier.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("AUC Score:", auc_score)

#plot 
plt.figure()
plt.plot(fpr, tpr, color = 'blue', label = 'ROC Curve (area = %0.2f)')
plt.plot([0,1],[0,1], color = 'grey', linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics (ROC)')
plt.legend(loc = 'lower right')
plt.grid()
plt.show()

# Future prediction
data1 = pd.read_csv(r"C:\Users\visha\OneDrive\Desktop\Data Science\Classroom\3.May 2025\8th May Logistic Regression\2.LOGISTIC REGRESSION CODE\final1.csv") 

d2 = data1.copy()

data1 = data1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(data1)

y_pred1 = pd.DataFrame()

d2['y_pred1'] = classifier.predict(M)

d2.to_csv('final2.csv')



















