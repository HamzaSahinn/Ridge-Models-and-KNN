# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:53:25 2020

@author: Abdullah Hamza Şahin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

data_set = load_boston()
X = data_set.data
y = data_set.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)
poly_2 = PolynomialFeatures(2,interaction_only=True)


X_train_poly2 = poly_2.fit_transform(X_train)
poly_2.fit(X_train_poly2, y_train)


X_test_poly2 = poly_2.fit_transform(X_test)
poly_2.fit(X_test_poly2, y_test)

alpha_list = np.linspace(0.0000001,500,50)
performance = []

for alpha in alpha_list:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    
    ridge_poly2_model = Ridge(alpha=alpha)
    ridge_poly2_model.fit(X_train_poly2, y_train)
    
    
    y_pred_ridge = ridge_model.predict(X_test)
    y_pred_poly2 = ridge_poly2_model.predict(X_test_poly2)
    
    rmse_ridge = (np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
    rmse_ridge_poly = (np.sqrt(mean_squared_error(y_test, y_pred_poly2)))
    performance.append((alpha, rmse_ridge, rmse_ridge_poly))

min_ridge = performance[0]
min_poly = performance[0]
for i in range(1,len(performance)) :
        
    if min_ridge[1] > performance[i][1]:
            min_ridge = performance[i];
    
    if min_poly[2] >performance[i][2]:
        min_poly = performance[i]

print("Alpha for linear ridge model =>",min_ridge[0],"  ", "ALpha for polynomial ridge model =>", min_poly[0] )
print("RMSE for ridge model = ",min_ridge[1])
print("RMSE for polynomial model = ",min_poly[2])
 
ridge_model = Ridge(alpha=min_ridge[0])
ridge_model.fit(X_train, y_train)

ridge_poly2_model = Ridge(alpha=min_poly[0])
ridge_poly2_model.fit(X_train_poly2, y_train)


y_pred_ridge = ridge_model.predict(X_test)
y_pred_poly2 = ridge_poly2_model.predict(X_test_poly2)

fig = plt.figure()


ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(y_test, y_pred_ridge)
ax1.set_xlabel("y_test")
ax1.set_ylabel("y_pred_ridge")
ax1.set_title("Linear Ridge")

ax2.scatter(y_test, y_pred_poly2)
ax2.set_xlabel("y_test")
ax2.set_ylabel("y_pred_poly")
ax2.set_title("Poly2 Ridge")

fig.tight_layout()

fig.savefig("ps1_1.png")







