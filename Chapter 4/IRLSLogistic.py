import numpy as np
import math
import matplotlib.pyplot as plt
import random

"""Implement Logistic Regression using Iterated Reweighted Least Square Algorithm"""
class IRLSLogistic:
    def __init__(self):
        self.weight=np.NaN
    
    def fit(self,x,t,numberIterations,w_initial):
        self.weight =self.IRLS(w_initial,x,t)
        for i in xrange (numberIterations):
            self.weight=self.IRLS(self.weight,x,t)
            self.weight=self.weight/np.linalg.norm(self.weight)
        return self 
        
    def IRLS(self,w_old,x,t):
        y=sigmoid(x.dot(w_old))
        "weightMatrix is R in book"
        weightMatrix=np.zeros((y.shape[0],y.shape[0]))
        for i in xrange (y.shape[0]):
            weightMatrix[i][i]=y[i]*(1-y[i])    
        hessian=x.T.dot(weightMatrix).dot(x)
        z=x.dot(w_old)-np.linalg.pinv(weightMatrix).dot(y-t)
        w_new=w_old-np.linalg.inv(hessian).dot(x.T).dot(weightMatrix).dot(z)
        return w_new
    
    def plotSeperationLine(self):
        x1_plot=np.arange(0.1,5.1,0.1).reshape(50,1)
        x2_plot=-x1_plot*self.weight[0]/self.weight[1]
        plt.plot(x1_plot,x2_plot)
        return self
            
"""Logistic Sigmoid Function for matrix"""
def sigmoid(x):
    x_sigmoid=np.ones((x.shape[0],x.shape[1]))
    for i in xrange (x.shape[0]):
        for j in xrange (x.shape[1]):
            x_sigmoid[i][j]=1 / (1 + math.exp(-x[i][j])) 
    return x_sigmoid.reshape(x.shape[0],x.shape[1])
 
"Generate and Plot Linearly Seperable Data"
"Blue represents class 0"
x1_0=np.arange(0.1,5.1,0.1).reshape(50,1)
x2_0=np.array([random.uniform(0,i) for i in x1_0])
x1_1=np.arange(0.1,5.1,0.1).reshape(50,1)
x2_1=np.array([random.uniform(i,5) for i in x1_1])
t_0=np.zeros((50,1))
t_1=np.ones((50,1))
"Blue represents class 0"
"Red represents class 1"
plt.scatter(x1_0,x2_0,color='blue')
plt.scatter(x1_1,x2_1,color='red')

"Prepare Data for Fitting"
x_0=np.concatenate((x1_0,x2_0),axis=1)
x_1=np.concatenate((x1_1,x2_1),axis=1)
x=np.concatenate((x_0,x_1),axis=0)
t=np.concatenate((t_0,t_1))

"Initialize a random weight vector"
w_initial=np.random.rand(2,1)
numberIterations=10

"Fit and plot the IRLS logistic regressor"
regressor=IRLSLogistic()
regressor.fit(x,t,numberIterations,w_initial)
regressor.plotSeperationLine()