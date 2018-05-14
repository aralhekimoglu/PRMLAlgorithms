import numpy as np
import matplotlib.pyplot as plt
import math

class Gaussian:
    def __init__(self,mean=np.array([0, 0]),covariance=[[1, 0], [0, 1]]):
        self.mean=mean.reshape(2,1)
        self.covariance=covariance
        self.dimension=2
    
    def calculateProbability(self,x):
        """
            Args:
                x (np.array) of size 2: input to calculate prob. for
            Returns:
                probabilty (float): calculated probability
        """
        x=x.reshape(2,1)
        sigma_inverse=np.linalg.inv(self.covariance)
        sigma_determinant=abs(np.linalg.det(self.covariance))
        divisor=1/(math.pow(2*math.pi,self.dimension/2)*math.pow(sigma_determinant,0.5))
        varX=np.array(x-self.mean)
        expo=math.exp(  (-0.5)*(varX.T.dot(sigma_inverse).dot(varX))  )
        return divisor*expo
    
    def plotGaussian(self,x1limit,x2limit,color='black'):
        """
            Args:
                x1limit (list) of size 2: vertical axis limits
                x2limit (list) of size 2: horizontal axis limits
                color (string): color for plotting
            Returns:
                an instance of self
        """
        amountData=100
        x1 = np.linspace(x1limit[0], x1limit[1], amountData)
        x2 = np.linspace(x2limit[0], x2limit[1], amountData)
        
        n=[]
        X, Y = np.meshgrid(x1, x2)
        for i in range(0,amountData):
            for j in range(0,amountData):
                inX=np.array([X[i][j],Y[i][j]]).reshape(2,1)
                n.append(self.calculateProbability(inX))
                
        outx=np.matrix(n).reshape(amountData,amountData)
        plt.contour(X, Y, outx, colors=color)
        plt.xlim(x1limit)
        plt.ylim(x2limit)
        return self