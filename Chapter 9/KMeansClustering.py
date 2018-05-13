import numpy as np
import matplotlib.pyplot as plt

class KMeansCluster:
    def __init__(self,K):
        self.K=K
        self.mean=np.random.uniform(low=-3.5, high=5.5, size=(K,2))
        self.J=[99999999999999]
        
    def plotCluster(self,cluster,color):
        cluster=np.array(cluster).reshape((len(cluster),2))
        scatterPlot2DMatrix(cluster,x1limit,x2limit,color)
    
    def closestMean(self,x,mean):
        indice=0
        minDist=euclidienDistance(x,mean[0])
        for i in range (mean.shape[0]-1):
            newDist=euclidienDistance(x,mean[i+1])
            if minDist>newDist:
                indice=i+1
        return indice,minDist

    def fit(self,tol=0.01,numIter=10,plotClusters=False):
        for j in range(numIter):
            J_new=0
            assignedClusters=[]
            
            for i in range (self.K):    
                assignedClusters.append([])
            
            for i in range(data.shape[0]):
                assignedMean,Ji=self.closestMean(data[i],self.mean)
                assignedClusters[assignedMean].append(data[i])
                J_new+=Ji
            if (abs((self.J[-1]-J_new)/J_new) < tol)  :
                break
            self.J.append(J_new)
            
            color=iter(plt.cm.rainbow(np.linspace(0,1,3)))
            for i in range(len(assignedClusters)):
                cluster=assignedClusters[i]
                cluster=np.array(cluster).reshape((len(cluster),2))
                self.mean[i]=cluster.sum(axis=0)/cluster.shape[0]
                if plotClusters:
                    plt.figure(j)
                    c=next(color)
                    scatterPlot2DMatrix(cluster,x1limit,x2limit,c)
            if plotClusters:   
                plt.scatter(self.mean[:,0],self.mean[:,1], marker='*', s=200, c='#050505')
        return assignedClusters
    
    def plotCostHistory(self):
        J=self.J[1:]
        x=[i for i in xrange (len(J))]
        plt.figure()
        plt.plot(x,J)
        plt.title("Cost History of Fitting")

"""Generates Gaussian Data for the given amount of data"""
def generateGaussianData(mean,covariance,amountData=100):
    return np.matrix([np.random.multivariate_normal(mean, covariance) for i in range (0,amountData)])

"""Plots a 2d scatter plot for matrix"""
def scatterPlot2DMatrix(data,x1limit,x2limit,pointsColor): 
    for i in range (0,data.shape[0]):
        plt.scatter(data[i,0],data[i,1],color=pointsColor)
    plt.xlim(x1limit)
    plt.ylim(x2limit)
    plt.show()
    
def euclidienDistance(x1,x2):
    return np.linalg.norm(x1-x2)
    
np.random.seed(6)
"Generated data"
mean_blue = np.array([0, 0])
cov_blue = [[1, 0], [0, 1]]
mean_red = np.array([-5, 5])
cov_red = [[1, 0], [0, 1]]
mean_green = np.array([4, 3])
cov_green = [[1, 0], [0, 1]]
data_blue=generateGaussianData(mean_blue,cov_blue)
data_red=generateGaussianData(mean_red,cov_red)
data_green=generateGaussianData(mean_green,cov_green)
data=np.concatenate((data_red,data_blue,data_green))
x1limit=[-8,7]
x2limit=[-4,9]

kmeanscluster=KMeansCluster(3)
clusteredData=kmeanscluster.fit(0.01,10,plotClusters=True)
kmeanscluster.plotCostHistory()

"Plot generated Data"
plt.figure()
plt.title('Generated Data')
scatterPlot2DMatrix(data,x1limit,x2limit,'gray')