#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Breast_cancer_data.csv.xls')
x=dataset.iloc[:,0:5].values
y=dataset.iloc[:,5].values

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(x_train, y_train)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

a=float(input("Enter mean radius: "))
b=float(input("Enter mean texture: "))
c=float(input("Enter mean perimeter: "))
d=float(input("Enter mean area: "))
e=float(input("Enter mean smoothness: "))

Xnew = [[a,b,c,d,e]]
# make a prediction

ynew = classifier.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
