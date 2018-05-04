import random
import numpy as np
import math
import matplotlib.pyplot as plt

class FisherLDA:
    def __init__(self):
        
        self._weightVector=np.NaN
        self._w0=0
    
    @property
    def weightVector(self):
        return self._weightVector
    
    @weightVector.setter
    def weightVector(self,value):
        self._weightVector=value
    
    @property
    def w0(self):
        return self._w0
    
    @w0.setter
    def w0(self,value):
        self._w0=value
    
    def fit(self,data1,data2):
        """
        Args:
            data1 (np.matrix) of shape[amountOfData1,2]: first data points each row a 2d point
            data2 (np.matrix) of shape[amountOfData2,2]: second data points each row a 2d point
        Returns:
            an instance of self
        """
        meanData1=data1.mean(0)
        dataMinusMean1=data1-meanData1 
        cov_matrix1=dataMinusMean1.T.dot(dataMinusMean1)
        meanData2=data2.mean(0)
        dataMinusMean2=data2-meanData2
        cov_matrix2=dataMinusMean2.T.dot(dataMinusMean2)
        
        S_w_inverse=np.linalg.inv(cov_matrix1+cov_matrix2)
        self.weightVector=S_w_inverse.dot((meanData1-meanData2).T)
        self.w0=-0.5*(meanData1.dot(S_w_inverse.dot(meanData1.T))-meanData2.dot(S_w_inverse.dot(meanData2.T)))+math.log(data1.shape[0]*1.0/data2.shape[0])
        return self
    
    def plotPrediction(self):
        """
        Args:
            none
        Returns:
            nothing
        """
        x2=np.arange(-3,5.2,0.1)
        x1=(-self.w0[0]-self.weightVector[1]*x2)/self.weightVector[0]
        x1=x1.reshape(x2.shape[0],1)
        plt.plot(x1,x2)
      
def generate_data(x_min,x_max,a,b,amount=100,noise=0.6):
    """
        Args:
            x_min(int): min of x1
            x_max(int): max of x1
            a (float): coeff. of x1 in linear function x2=a*x+b
            b (float): bias of linear function x2=a*x+b
            amount (int): amount of (x1,x2) points to be generated
            noise (float): amount of gaussian noise to be added
        Returns:
            generatedData (matrix) of shape [amount,2]: each row is 2d point (x1,x2)
        """
    f=linear_function(a,b)
    x1=[random.uniform(x_min,x_max) for i in range(0,amount)]
    x2=[f(i) for i in x1]
    x2_noised=[np.random.normal(i,noise) for i in x2]
    return np.matrix([x1,x2_noised]).T

"""Returns y=a*x+b """
def linear_function (a,b): return lambda x: float(a)*x+b 

"Plots a 2d scatter plot for matrix"
def scatterPlot2DMatrix(data,x1limit,x2limit,pointsColor): 
    for i in range (0,data.shape[0]):
        plt.scatter(data[i,0],data[i,1],color=pointsColor)
    plt.xlim(x1limit)
    plt.ylim(x2limit)
    
"""Generate data for blue and red points"""
data_blue=generate_data(-2,4,a=0.3,b=3,amount=200,noise=0.6)
data_red=generate_data(2,7,a=0.4,b=-1,amount=200,noise=0.6)

"""Plot data for blue and red points"""
scatterPlot2DMatrix(data_blue,[-3,7],[-3,5],'blue')
scatterPlot2DMatrix(data_red,[-3,7],[-3,5],'red')

"""Fit and plot the prediction line"""
LDA=FisherLDA()
LDA.fit(data_blue,data_red)
LDA.plotPrediction()