import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier as knn

data=pd.read_csv('iris.csv')
x=np.array(data.iloc[:,:-1])
y=np.array(data.iloc[:,-1])
xtr,xt,ytr,yt=train_test_split(x,y,test_size=.5)
clf=knn(n_neighbors=3)
clf.fit(xtr,ytr)
y_pred=clf.predict(xt)
n=len(xtr)
for i in range(0,n):
	if yt[i]==y_pred[i]:
		print(yt[i]," is correctly predicted as: ",y_pred[i])
	else:
		print(yt[i]," is wrongly predicted as: ",y_pred[i])
print("accuracy= ",accuracy_score(yt,y_pred))
print("confusion matrix:\n",confusion_matrix(y_pred,yt))
