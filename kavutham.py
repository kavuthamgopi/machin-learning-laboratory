import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('C:/Users/RIT/PycharmProjects/Descision trees/Social_Network_Ads.csv')
print(data.head())
feature_cols=['Age','EstimatedSalary']
X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier()
classifier = classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
from sklearn import metrics
print('Accuracy Score:',metrics.accuracy_score(Y_test,Y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)




