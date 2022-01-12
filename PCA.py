# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:32:50 2021

@author: suhed
"""

import numpy as np


def inputData(column, row):
    data = [[float((input("[" +str(r) +"] [" +str(c) +"] : "))) for c in range(column)] for r in range(row)]
    return np.array(data)

def dataMean(data):
    mean = []
    for i in range(len(data[0])):
        mean.append(sum(data[:,i])/len(data))
    return np.array(mean)

def dataVar(data):
    mean = dataMean(data)
    total = 0
    for x in data:
        total += pow(sum(x - mean),2)
    return (total / (len(data) - 1))
    
def dataCov(x,y):
    xM, yM = x.mean(), y.mean()
    return sum((x-xM) * (y - yM)) / (len(x) - 1)


def dataCovMa(data):
    return np.array([[dataCov(data[:,i], data[:,j]) for j in range(len(data[0]))] for i in range(len(data[0]))])
            

#data = inputData(2, 10)

#mean = dataMean(data)
#std = pow(dataVar(data), 1/2)
#var = dataVar(data)
#CovMat = dataCovMa(data)


