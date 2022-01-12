# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 22:52:03 2022

@author: suhed
"""

from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog as fd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import matplotlib.pyplot as plt
import numpy as np
import PCA
import KMeans
import KNN
import BasicLinearRegression as blr

def open():
    filename = fd.askopenfilename(title = "Select a file",initialdir = os.getcwd(), filetypes = (("csv Files", "*.csv"),
                                                                         ("All Files", "*.*")))


class PCAWindow(Toplevel):
    
    def __init__(self, master = None):
        super().__init__(master = master)
        self.filename = 0
        self.data = 0
        self.title("PCA")
        self.geometry("600x600")
        label = Label(self, text = "This is PCA Window")
        label.pack()
        self.fileButton()
        writeButton = Button(self, text = "PCA Results", command = self.PCAResults)
        writeButton.pack()
        
        
        
        
    def openfile(self):
        self.filename = fd.askopenfilename(title = "Select a file",initialdir = os.getcwd(), filetypes = (("csv Files", "*.csv"),
                                                                                                     ("All Files", "*.*")))
    
    def fileButton(self):
        filebtn = Button(self, text = "Open File", command=self.openfile)
        filebtn.pack()        
        
    
    def PCAResults(self):
        self.data = np.genfromtxt(self.filename, delimiter=',', skip_header=1)
        text1 = "Data Means \n " + np.array_str(PCA.dataMean(self.data)) + "\nData Var \n" + str(PCA.dataVar(self.data)) + "\nDava Cov Matrix \n " + np.array_str(PCA.dataCovMa(self.data))
        label1 = Label(self, text = text1)
        label1.pack()

class KMeansWindow(Toplevel):
    
    def __init__(self, master = None):
        super().__init__(master = master)
        self.filename = 0
        self.data = 0
        self.numberCluster = 0
        self.title("KMeans")
        self.geometry("600x600")
        label = Label(self, text = "This is Kmeans Window")
        label.pack()
        self.fileButton()
        label = Label(self, text = "Enter Number of Iteration")
        label.pack()
        self.eIte = Entry(self, width = 30)
        self.eIte.pack()
        label = Label(self, text = "Enter Number of Cluster")
        label.pack()
        self.eC = Entry(self, width = 30)
        self.eC.pack()
        label = Label(self, text = "Select Distance Function")
        label.pack()
        self.dist = StringVar()
        disChoosen = Combobox(self, width = 30, textvariable = self.dist)
        disChoosen['values'] = ('minkowski',
                                'manhattan',
                                'eucledian')
        disChoosen.current(2)
        disChoosen.pack()
        
        self.runButton()
    
    def fileButton(self):
        filebtn = Button(self, text = "Open File", command=self.openfile)
        filebtn.pack()   

    def openfile(self):
        self.filename = fd.askopenfilename(title = "Select a file",initialdir = os.getcwd(), filetypes = (("csv Files", "*.csv"),
                                                                                                     ("All Files", "*.*")))
    def runButton(self):
        runBtn = Button(self, text = "RUN", command=self.runKmeans)
        runBtn.pack()
    
    def runKmeans(self):
        self.data = np.genfromtxt(self.filename, delimiter=',', skip_header=1)
        clus = KMeans.KMeans(n_cluster=int(self.eC.get()), max_iter=int(self.eIte.get()), distance_fun = self.dist.get())    
        clusters = clus.fit(self.data)
        #plot Kmeans
        colors = list("rgbcmyk")
        for key in clus.clusters:
            for value in clus.clusters[key]:
                plt.scatter(value[0], value[1] , color = colors[key])
        plt.show()



class KNNWindow(Toplevel):
    def __init__(self, master = None):
        super().__init__(master = master)
        self.filename = 0
        self.data = 0
        self.title("KNN")
        self.geometry("600x600")
        label = Label(self, text = "This is KNN Window")
        label.pack()
        self.fileButton()
        label = Label(self, text = "Enter 'k' Value")
        label.pack()
        self.eK = Entry(self, width = 30)
        self.eK.pack()
        label = Label(self, text = "Enter 'x' Value")
        label.pack()
        self.eX = Entry(self, width = 30)
        self.eX.pack()
        label = Label(self, text = "Enter 'y' Value")
        label.pack()
        self.eY = Entry(self, width = 30)
        self.eY.pack()
        self.runButton()
        
        
        
    def fileButton(self):
        filebtn = Button(self, text = "Open File", command=self.openfile)
        filebtn.pack()   

    def openfile(self):
        self.filename = fd.askopenfilename(title = "Select a file",initialdir = os.getcwd(), filetypes = (("csv Files", "*.csv"),
                                                                                                     ("All Files", "*.*")))
    
    def runButton(self):
        runBtn = Button(self, text = "RUN", command=self.runKNN)
        runBtn.pack()        
        
    def runKNN(self):
        self.data = np.genfromtxt(self.filename, delimiter=',', skip_header=1)
        x_test = np.array([int(self.eX.get()), int(self.eY.get())])
        y_train = self.data[:,-1]
        y_train = y_train[..., None]
        x_train = self.data[:,:-1]
        knn = KNN.KNN(int(self.eK.get()))
        knn.fit(x_train, y_train)
        y_test = knn.predict(x_test)
        Tx = "Prediction Result : " + str(y_test)
        label = Label(self, text = Tx).pack()


class LinRegWindow(Toplevel):
    
    def __init__(self, master = None):
        super().__init__(master = master)
        self.filename = 0
        self.data = 0
        self.title("Linear Regression")
        self.geometry("600x600")
        label = Label(self, text = "This is Linear Regression Window")
        label.pack()
        self.fileButton()
        self.runButton()
        

    def fileButton(self):
        filebtn = Button(self, text = "Open File", command=self.openfile)
        filebtn.pack()   

    def openfile(self):
        self.filename = fd.askopenfilename(title = "Select a file",initialdir = os.getcwd(), filetypes = (("csv Files", "*.csv"),
                                                                                                     ("All Files", "*.*")))
    def runButton(self):
        runBtn = Button(self, text = "RUN", command=self.runKNN)
        runBtn.pack()        
        
    def runKNN(self):
        self.data = np.genfromtxt(self.filename, delimiter=',', skip_header=1)
        baLiRe = blr.BasicLinearReg()
        baLiRe.fit(self.data)
        y = baLiRe.b * self.data[:,0] + baLiRe.a + baLiRe.e
        y1 = baLiRe.b * self.data[:,0] + baLiRe.a

        plt.plot(self.data[:,0], y, '-r', label = 'y = bx+a+e')
        plt.plot(self.data[:,0], y1, '--b', label = 'y = bx+a')
        plt.plot(self.data[:,0], self.data[:,1], 'o', label = 'x,y')
        plt.legend()
        plt.show()
        
master = Tk()
master.title('Patern Recognition')


master.geometry("600x600")




label = Label(master, text = "This is main Window")
label.pack(side = TOP, pady = 1)
#btn1 = Button(master, text = "PCA1")

btn = Button(master, text = "PCA")
btn.bind("<Button>", 
         lambda e: PCAWindow(master))
btn.pack(pady=10)


btn1 = Button(master, text = "KMeans")
btn1.bind("<Button>", 
         lambda e: KMeansWindow(master))
btn1.pack(pady=10)


btn2 = Button(master, text = "KNN")
btn2.bind("<Button>", 
         lambda e: KNNWindow(master))
btn2.pack(pady=10)

btn3 = Button(master, text = "Linear Regression")
btn3.bind("<Button>", 
         lambda e: LinRegWindow(master))
btn3.pack(pady=10)
master.mainloop()