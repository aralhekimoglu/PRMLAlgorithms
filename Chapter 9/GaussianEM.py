import numpy as np
import random
import matplotlib.pyplot as plt
import math
from Gaussian import Gaussian
from Utils import scatterPlot2DMatrix,generateGaussianData

class GaussianEM:
    def __init__(self,data,K):
        self.data=data
        self.N=data.shape[0]
        self.K=K
        self.respon=np.zeros((self.N,self.K))
        self.cov=[np.random.uniform(low=0, high=1)*np.identity(K) for i in range (K)]
        self.mean=np.random.uniform(low=-3.5, high=5.5, size=(K,2))
        self.mixCoeff=[1.0/K for i in range (K)]
        
    def E_step(self):
        for n in range (self.N):
            temp=np.zeros((1,self.K))
            for k in range (self.K):       
                cov_in=np.array([self.cov[k][0],self.cov[k][1]])
                gaussian=Gaussian(self.mean[k],cov_in)
                temp[0,k]=self.mixCoeff[k]*gaussian.calculateProbability(self.data[n])
            self.respon[n]=[temp[0,k]/temp.sum() for k in range(self.K)]
        return self.respon
    
    def M_step(self):
        
        Nk=self.respon.sum(axis=0)
        self.mixCoeff=Nk/Nk.sum()
        
        for k in range (self.K):
            responArray=self.respon[:,k].reshape(self.N,1)
            responMatrix=np.concatenate((responArray,responArray),axis=1)
            data_responed=np.multiply(responMatrix,self.data)
            self.mean[k]=(1/Nk[k])*np.sum(data_responed,axis=0)

        for k in range(self.K):
            responArray=self.respon[:,k].reshape(self.N,1)
            responMatrix=np.concatenate((responArray,responArray),axis=1)
            data_mean=self.data-self.mean[k]
            data_mean_rr=np.multiply(data_mean,responMatrix)
            self.cov[k]=(1/Nk[k])*data_mean_rr.T.dot(data_mean)
            self.cov[k]=np.array([ [math.sqrt(self.cov[k][0,0]), 0], [ 0, math.sqrt(self.cov[k][1,1]) ] ])
        
        return self.mean,self.cov,self.mixCoeff
    
    def fit(self,numIters=5,tol=0.1):
        for iterations in range(numIters):
            oldMean=self.mean.copy()
            self.E_step()
            self.M_step()
            if (self.meanChange(oldMean)<tol):
                break
    
    def meanChange(self,oldMean):
        return np.linalg.norm(oldMean-self.mean)
                
    def plotGaussians(self,colors=['blue','red']):
        for k in range (self.K):
            x_min, x_max = self.data[:, 0].min() - .5, self.data[:, 0].max() + .5
            y_min, y_max = self.data[:, 1].min() - .5, self.data[:, 1].max() + .5
            gaussian=Gaussian(self.mean[k],self.cov[k])
            gaussian.plotGaussian([x_min,x_max],[y_min,y_max],colors[k])

random.seed(3)
np.random.seed(10)

mean_blue=np.random.uniform(low=-5, high=5, size=(2,))
cov_blue = [[random.randint(1,4), 0], [0, random.randint(1,4)]]
mean_red=np.random.uniform(low=-5, high=5, size=(2,))
cov_red = [[random.randint(1,4), 0], [0, random.randint(1,4)]]
data_blue=generateGaussianData(mean_blue,cov_blue)
data_red=generateGaussianData(mean_red,cov_red)

data=np.concatenate((data_red,data_blue))

"Plot generated Data"
plt.figure()
plt.title('Data')
scatterPlot2DMatrix(data,'gray')

gaussianEM=GaussianEM(data,K=2)
gaussianEM.fit()
gaussianEM.plotGaussians()
