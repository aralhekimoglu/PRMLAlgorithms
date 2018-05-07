import numpy as np
import math
import matplotlib.pyplot as plt
import random

"""Implement Logistic Regression using Gradient Descent"""
class LogisticRegressor:
    def __init__(self):
        self.weights=np.NaN
        self.alpha=0
        self.costHistory=np.NaN
        
    def calculateY(self,x):
        z = np.zeros((1,1))
        for i in xrange(len(self.weights)):
            z += x[i]*self.weights[i]
        return sigmoid(z)
    
    def errorFunction(self,x,t):
        errorSum = 0
        for i in xrange(x.shape[0]):
            yn = self.calculateY(x[i])
            errorSum+=t[i] * math.log(yn)+(1-t[i]) * math.log(1-yn)
        return -errorSum
    
    def gradientError(self,x,t):
        errorVector = np.array((1,x.shape[1]))
        for i in xrange(x.shape[0]):
            yn = self.calculateY(x[i])
            errorVector = errorVector+(yn - t[i])*x[i]         
        return errorVector.reshape(2,1)
    
    def gradientDescent(self,x,t):
        w_new=np.array((2,1))
        gradError = self.gradientError(x,t)
        self.weights =  self.weights- (self.alpha/x.shape[0])*gradError
        return w_new
    
    def fit(self,x,t,alpha,weights,numberIterations):
        self.weights=weights
        self.alpha=alpha
        self.costHistory=np.zeros((numberIterations,1))
        for i in xrange(numberIterations):
            self.gradientDescent(x,t)
            self.costHistory[i]=(self.errorFunction(x,t))
        return self
   
    def plotSeperationLine(self):
        x1_plot=np.arange(0.1,5.1,0.1).reshape(50,1)
        x2_plot=-x1_plot*self.weights[0]/self.weights[1]
        plt.plot(x1_plot,x2_plot)
        return self
    
    def plotCostHistory(self,numberIterations):
        x=np.arange(0,numberIterations,1)
        y=self.costHistory
        plt.figure(2)
        plt.plot(y,x)
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

"Initialize a random weight vector and parameters"
w_initial=np.random.rand(2,1)
alpha = 0.1
numberIterations = 200

"Fit and plot the Logistic regressor with gradient descent"
regressor=LogisticRegressor()
regressor.fit(x,t,alpha,w_initial,numberIterations)
regressor.plotSeperationLine()
regressor.plotCostHistory(numberIterations)