#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:20:15 2021

@author: semihacmali
"""

import numpy as np
from collections import Counter

def inputData(column, row):
    data = [[int((input("[" +str(r) +"] [" +str(c) +"] : "))) for c in range(column)] for r in range(row)]
    return np.array(data)

def distance(X1, X2, p = 2):
    X1 = np.array(X1)
    X2 = np.array(X2)
    return pow(sum(pow(abs(X1-X2),p)), 1/p)



    

class KNN:
    def __init__(self, k):
        self.k = k
        
    
    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y
        
    def predict(self, X_test):
        out = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(self.X_train)):
                dist = distance(self.X_train[j], X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(self.Y_train[j][0])
            ans = Counter(votes).most_common(1)[0][0]
            out.append(ans)
        return out
    
    
x_train = inputData(2,12)
y_train = inputData(1,12)

x_test = inputData(2,1)


knn = KNN(4)
knn.fit(x_train, y_train)
y_test = knn.predict(x_test)


