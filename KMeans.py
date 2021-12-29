# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random

def inputData(column, row):
    data = [[int((input("[" +str(r) +"] [" +str(c) +"] : "))) for c in range(column)] for r in range(row)]
    return np.array(data)




class KMeans():
    
    def __init__(self, n_cluster = 3, max_iter = 10, distance_fun = "eucledian"):
        self.k = n_cluster
        self.iter = max_iter
        if(distance_fun == "minkowski"):
            self.power = 3
        elif(distance_fun == "manhattan"):
            self.power = 1
        else:
            self.power = 2
        self.centroids = {}
        self.clusters = {}
    
    def distance(self, X1, X2):
        X1 = np.array(X1)
        X2 = np.array(X2)
        return pow(sum(pow(abs(X1-X2),self.power)), 1/self.power)
    

    def recalculate_centroids(self):
        for i in range(self.k):
            self.centroids[i] = np.average(self.clusters[i], axis = 0)
        return self.centroids
    
    def recalculate_clusters(self, data):
        
        #clusters = {}
        
        for i in range(self.k):
            self.clusters[i] = []
        for X in data:
            euc_dis = []
            for j in range(self.k):
                euc_dis.append(self.distance(X, self.centroids[j]))
            self.clusters[euc_dis.index(min(euc_dis))].append(X)
        
        return self.clusters
    
    
    def fit(self, data):
        #clusters = {}
        #centroids = {}
        first_index = random.sample(range(0,len(data)),self.k)
        for i in range(self.k):
            self.clusters[i] = []
            self.centroids[i] = first_index[i]
            
        for X in data:
            euc_dis = []
            for j in range(self.k):
                euc_dis.append(self.distance(X, self.centroids[j]))
            self.clusters[euc_dis.index(min(euc_dis))].append(X)
        
        for i in range(1, self.iter):
            self.clusters = self.recalculate_clusters(data)
            nwcen = self.recalculate_centroids()
            if(nwcen == self.centroids):
                return self.clusters
            else:
                self.centroids = nwcen
        
        return self.clusters
    

data = inputData(2, 4)

clus = KMeans(n_cluster=2, max_iter=5, distance_fun = "eucledian")    

clusters = clus.fit(data)            

clus.centroids
clus.clusters
