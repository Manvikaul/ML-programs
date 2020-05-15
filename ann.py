#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#encoding categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1=LabelEncoder()
x[:,1]=labelencoder_x1.fit_transform(x[:,1])

labelencoder_x2=LabelEncoder()
x[:,2]=labelencoder_x2.fit_transform(x[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
#initialising ANN
classifier=Sequential()
#1st hidden layer and input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#2nd hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',mertics=['accuracy'])
#fit ANN to training set
classifier.fit(x_train,y_train,batch_size=10,epochs=100)
#predicting test results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




