#importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#datapre processing 
#1 importing dataset 
dataset = pd.read_csv('Bank_Data.csv')
X = dataset.iloc[:,3:13].values #significant independent variables
y = dataset.iloc[:,13].values  
print(X)              #categorical data country and gender needs to be encoded ..consider all attributes do not ignore significant inputs
#2 Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder # encode label and encode data with more than one column to required size
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
print(X)
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
print(X)
#0,1,2 should not be mistaken to priority hence use onehotencoder on country column
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
print(X)
#in order to avoid randomtrap or corelation error remove one column
X = X[:,1:]  #to avoid dummy variable trap 
X
# To encode the output data( if it is categorical)
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#feature scaling is mandatory for neural networks as it processes large amounts of data eg credit score and estimated salary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Now let's make the ANN
#importing keras libraries and packages 
import keras 
from keras.models import Sequential
from keras.layers import Dense #how many layers are there
#initializing the neural network
classifier = Sequential()
#adding the input and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11)) #init -> intialize weights method ...initialize weights with smaller number equal to 0
#adding the second layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#adding the output layer 
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting the ANN to the training set in batch of 10 training data
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#prediciting the test set results 
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) # chance of leaving bank more than half 

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)