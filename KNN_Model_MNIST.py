# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:33:30 2020

@author: Lenovo
"""


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


data_set = datasets.load_digits()

(X_train, X_test, y_train, y_test) = train_test_split(np.array(data_set.data),data_set.target, test_size=0.2, random_state=41)


acc = []
for k in range(1, 21):
        
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    acc.append(score)
    print("K=",k,"  acc: %",score)

index = np.argmax(acc)
k = index+1

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.scatter(predictions,y_test)
plt.xlabel("Predictions")
plt.ylabel("True values")
plt.title("KNN (K=1)")
plt.savefig("KNN.png")
