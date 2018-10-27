# -*- coding: utf-8 -*-
"""
Spyder Editor

@author:Himanshu
"""
#Importing the required libraries
import keras 
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#Importing the dataset
dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values
#Encoding the variables 
labelencoder_X_1=LabelEncoder()
labelencoder=LabelEncoder()
X[:,2]=labelencoder.fit_transform(X[:,2])
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:] 
#Splitting the dataset into the training set and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Feature Scaling the variables
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#Initialising the ANN
classifier=Sequential()
#Adding input and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#Adding the second hidden layer 
classifier.add(Dense(output_dim=11,init='uniform',activation='relu'))
#Adding the output_layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
#Predicting the test set result
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
#Confusing matrix
cm=confusion_matrix(y_test,y_pred)
    