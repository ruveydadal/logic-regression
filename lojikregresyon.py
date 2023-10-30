# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:11:39 2021

@author: muham
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

#CONFUSSION MATRIX
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

#K-NEAREST NEIGHBORHOOD
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')
#n_neighbors değeri komşu sayısıdır , veri kümesine göre değişir.
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)

#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
svc=SVC(kernel='linear')
#kernel değiştirilebilir
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)


#GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)

#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
#entropy'e göre hesaplandı , farklı şekilde de hesaplanabilir
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

#ROC, TPR,FPR DEĞERLERİ
y_proba=rfc.predict_proba(X_test)#olasılıkları hesaplar
print(y_proba[:,0])


from sklearn import metrics
fpr,tpr,thold=metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)
