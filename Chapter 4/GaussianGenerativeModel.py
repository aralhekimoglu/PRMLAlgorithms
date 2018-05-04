import numpy as np
import math
import matplotlib.pyplot as plt

class Gaussian:
    def __init__(self,mean=np.array([0, 0]),covariance=[[1, 0], [0, 1]]):
        self.mean=mean.reshape(2,1)
        self.covariance=covariance
        self.dimension=mean.shape[0]
    
    def calculateProbability(self,x):
        """
            Args:
                x (np.array) of size 2: input to calculate prob. for
            Returns:
                probabilty (float): calculated probability
        """
        x.reshape(2,1)
        sigma_inverse=np.linalg.inv(self.covariance)
        sigma_determinant=np.linalg.det(self.covariance)
        divisor=1/(math.pow(2*math.pi,self.dimension/2)*math.pow(sigma_determinant,0.5))
        varX=np.array(x-self.mean)
        expo=math.exp((-0.5)*(varX.T.dot(sigma_inverse).dot(varX)))
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

class GaussianGenerativeModel:
    def __init__(self):
        self.priorC1=0
        self.priorC2=0
        self.learned=False
        self.shared_cov=np.NaN
        self.mean1=np.NaN
        self.mean2=np.NaN
        self._weightVector=np.NaN
        self._w0=0
        self.gaussian1=Gaussian()
        self.gaussian2=Gaussian()
        
    def fit(self,data1,data2):
        """
            Args:
                data1 (np.matrix) of shape [amount1,2]
                data2 (np.matrix) of shape [amount2,2]
            Returns:
                an instance of self
        """
        amount1=data1.shape[0]
        amount2=data2.shape[0]
        self.priorC1=float(amount1)/(amount1+amount2)
        self.priorC2=1-self.priorC1
        
        self.mean1=np.mean(data1,axis=0)
        self.mean2=np.mean(data2,axis=0)
        
        var1=data1-self.mean1
        var1_t=var1.T
        cov1=var1_t.dot(var1)/amount1
                       
        var2=data2-self.mean2
        var2_t=var2.T
        cov2=var2_t.dot(var2)/amount2
        
        self.shared_cov=(amount1*cov1+amount2*cov2)/(amount1+amount2)
        self.gaussian1=Gaussian(self.mean1,self.shared_cov)
        self.gaussian2=Gaussian(self.mean2,self.shared_cov)
        shared_cov_inv=np.linalg.inv(self.shared_cov)
        self.weightVector=shared_cov_inv.dot((self.mean1-self.mean2).T)
        w0part1=(-0.5)*(self.mean1.dot(shared_cov_inv).dot(self.mean1.T))
        w0part2=(-0.5)*(self.mean2.dot(shared_cov_inv).dot(self.mean2.T))
        w0part3=np.log(self.priorC1/self.priorC2)
        self.w0=w0part1-w0part2+w0part3
        self.learned=True
        return self
    
    def plotGaussians(self,x1limit,x2limit):
        """
            Args:
                x1limit (list) of size 2: vertical axis limits
                x2limit (list) of size 2: horizontal axis limits
            Returns:
                an instance of self
        """
        self.gaussian1.plotGaussian(x1limit,x2limit,color='blue')
        self.gaussian2.plotGaussian(x1limit,x2limit,color='red')
        return self
        
    def plotPredictionLine(self):
        """
            Args:
                none
            Returns:
                an instance of self
        """
        x2=np.arange(-10,10,0.1)
        x1=(-self.w0[0]-self.weightVector[1]*x2)/self.weightVector[0]
        x1=x1.reshape(x2.shape[0],1)
        plt.plot(x1,x2)
        return self
    
    def predictClass(self,x):
        """
            Args:
                x (np.array) of size 2, will be reshaped to 2,1
            Returns:
                1: for class 1
                2: for class 2
        """
        x.reshape(2,1)
        probC1=sigmoid(self.weightVector.T.dot(x)+self.w0)
        if probC1>0.5:
            return 1
        else:
            return 2
        
"""Logistic Sigmoid Function"""
def sigmoid(x):
  return 1 / (1 + math.exp(-x))    
        
"""Plots a 2d scatter plot for matrix"""
def scatterPlot2DMatrix(data,x1limit,x2limit,pointsColor): 
    for i in range (0,data.shape[0]):
        plt.scatter(data[i,0],data[i,1],color=pointsColor)
    plt.xlim(x1limit)
    plt.ylim(x2limit)
    
"""Generates Gaussian Data for the given amount of data"""
def generateGaussianData(mean,covariance,amountData=100):
    return np.matrix([np.random.multivariate_normal(mean, covariance) for i in range (0,amountData)])

"""Generate Data"""
mean_blue = np.array([0, 0])
cov_blue = [[1, 0], [0, 1]]
mean_red = np.array([-5, 5])
cov_red = [[1, 0], [0, 1]]
data_blue=generateGaussianData(mean_blue,cov_blue)
data_red=generateGaussianData(mean_red,cov_red)

"""Plot generated Data"""
x1limit=[-8,3]
x2limit=[-4,9]
scatterPlot2DMatrix(data_red,x1limit,x2limit,'red')
scatterPlot2DMatrix(data_blue,x1limit,x2limit,'blue')
data1=data_blue
data2=data_red

"""Fit Model and Plot"""
gaussianGenerativeModel=GaussianGenerativeModel()
gaussianGenerativeModel.fit(data_blue,data_red)
gaussianGenerativeModel.plotGaussians(x1limit,x2limit)
gaussianGenerativeModel.plotPredictionLine()
prediction = gaussianGenerativeModel.predictClass(np.array([0.5,0.5]))