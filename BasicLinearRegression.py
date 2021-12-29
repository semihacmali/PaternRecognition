# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: semihacmali
"""
import numpy as np


def inputData(column, row):
    data = [[int((input("[" +str(r) +"] [" +str(c) +"] : "))) for c in range(column)] for r in range(row)]
    return np.array(data)


    

class BasicLinearReg:
    def __init__(self):
        
        self.a = 0
        self.b = 0
        self.e = 0
    
    def fit(self,data):
        x = data[:,0]
        y = data[:,1]
        self.b = (sum(x * y) - (len(data) * x.mean() * y.mean())) / (sum(x*x) - len(x) * pow(x.mean(),2))
        self.a = y.mean() - (x.mean() * self.b)
        self.calError(x, y)
        
    def predict(self, x):
        return x*self.b + self.a + self.e
    
    def calError(self, x, y):
        ypre = self.predict(x)
        self.e = pow(sum(pow(y-ypre,2)) / (len(y) -2), 1/2)
        
        
        
                
    



data = inputData(2, 5)

baLiRe = BasicLinearReg()
baLiRe.fit(data)
baLiRe.predict(7)
