import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from six import StringIO
import pydotplus
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
data=pd.read_csv('C:/Users/RIT/PycharmProjects/Descision trees/Social_Network_Ads.csv')
print(data.head())
feature_cols=['Age','EstimatedSalary']
X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values


from sklearn.model_selection \
import train_test_split
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
from matplotlib.colors import ListedColormap
X_set,Y_set = X_test,Y_test
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop= X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop= X_set[:,1].max()+1,step= 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],c=ListedColormap(("red","green"))(i),label=j)
plt.title("Decision Tree(Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated salary")
plt.legend()
plt.show()


dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.write_png("Desicion_Tree.png"))




