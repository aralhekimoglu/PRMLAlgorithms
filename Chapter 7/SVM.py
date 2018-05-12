import numpy as np
import random
import sklearn.datasets
import matplotlib.pyplot as plt
import math

class SVM:
    def __init__(self,X,y):
        self.b=0
        self.alpha=np.zeros((X.shape[0]))
        self.X=X
        self.y=y
        self.C=1

    def fit(self,C=1.0,tol=0.001,max_passes=10):
        passes=0
        self.C=C

        while(passes<max_passes):
            num_changed_alphas=0
            for i in xrange(X.shape[0]):
                
                K_i=self.computeKernelArray(self.X,self.X[i])
                f=np.sign((self.alpha * self.y).dot(K_i)+self.b)
                E_i = f - self.y[i]
                
                if (self.y[i]*E_i<-tol and self.alpha[i]<C) or ( self.y[i]*E_i>tol and self.alpha[i]>0):
                    j = self.randomIntNotZ(0, self.X.shape[0]-1, i) # Get random int j~=i
                    K_j=self.computeKernelArray(self.X,self.X[j])
                    f_j=np.sign((self.alpha * self.y).dot(K_j)+self.b)
                    E_j = f_j - self.y[j]
                    alpha_i_old=self.alpha[i]
                    alpha_j_old=self.alpha[j]
                    (L, H) = self.compute_L_H(alpha_i_old, alpha_j_old, self.y[i], self.y[j])
                    if(L==H):
                        continue
                    eta = self.kernelFunction(self.X[i], self.X[i]) + self.kernelFunction(self.X[j], self.X[j]) - 2 * self.kernelFunction(self.X[i], self.X[j])
                    if (eta<=0):
                        continue
                    alpha_unclipped = self.alpha[j] + float(self.y[j] * (E_i-E_j))/eta
                    self.alpha[j]=self.clip(alpha_unclipped,L,H)
                    if( abs(self.alpha[j]-alpha_j_old)<math.pow(10,-5) ):
                        continue
                    self.alpha[i] = self.alpha[i] + self.y[i]*self.y[j] * (alpha_j_old - self.alpha[i])
                    b1=self.b-E_i-self.y[i]*(self.alpha[i]-alpha_i_old)*K_i[i]-self.y[i]*(self.alpha[j]-alpha_j_old)*K_j[i]
                    b2=self.b-E_j-self.y[i]*(self.alpha[i]-alpha_i_old)*K_j[i]-self.y[j]*(self.alpha[j]-alpha_j_old)*K_j[j]
                    self.b=(b1+b2)/2
                    passes+=1
                    num_changed_alphas+=1
            if num_changed_alphas==0:
                passes+=1
            else:
                passes=0
       
    def randomIntNotZ(self,a,b,z):
        i = random.randint(a,b)
        while i == z:
            i = random.randint(a,b)
        return i
    
    def kernelFunction(self,x1, x2):
        return np.dot(x1, x2.T)
    
    def computeKernelArray(self,X,x):
        K=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            K[i]=self.kernelFunction(X[i],x)
        return K
    
    def clip(self,value,L,H):
        if(value>H):
            value=H
            return value
        elif(value<L):
            value=L
            return value
        else:
            return value        
    def compute_L_H(self, alpha_i_old, alpha_j_old, y_i, y_j):
        if(y_i == y_j):
            return (max(0, alpha_j_old + alpha_i_old - self.C), min(self.C, alpha_j_old + alpha_i_old))
        else:
            return (max(0, alpha_i_old - alpha_j_old), min(self.C, self.C - alpha_j_old + alpha_i_old))
     
    def predictFunction(self):
        return lambda x:(self.alpha * self.y).dot(self.computeKernelArray(self.X,x))+self.b
    
    def plotPredictionCurve(self):
        function=self.predictFunction()
        x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
        y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
        hh = 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hh), np.arange(y_min, y_max, hh))
        Z=np.zeros(xx.shape)
        for i in xrange(xx.shape[0]):
            for j in xrange(xx.shape[1]):
                al=np.array([xx[i][j], yy[i][j] ])
                pred=function(al)
                if(pred>=0):
                    Z[i][j]=1
                else:
                    Z[i][j]=0
        plt.contourf(xx, yy, Z,alpha=.8, cmap=plt.cm.RdBu)
        plt.title("Decision Boundary for Given Function")
        plt.show()

"Plots a 2d scatter plot for matrix"
def scatterPlot2DMatrix(data,x1limit,x2limit,pointsColor): 
    for i in range (0,data.shape[0]):
        plt.scatter(data[i,0],data[i,1],color=pointsColor)
    plt.xlim(x1limit)
    plt.ylim(x2limit)

def plotClassificationData(data_x,data_y):
    x_red=[]
    x_blue=[]
    for i in xrange (data_y.size):
        if data_y[i]==0:
            data_y[i]=-1
            x_red.append(data_x[i])
        else:
            x_blue.append(data_x[i])
    x_red=np.array(x_red)
    x_blue=np.array(x_blue)
    x_min, x_max = data_x[:, 0].min() - .5, data_x[:, 0].max() + .5
    y_min, y_max = data_x[:, 1].min() - .5, data_x[:, 1].max() + .5
    scatterPlot2DMatrix(x_blue,[x_min,x_max],[y_min,y_max],'blue')
    scatterPlot2DMatrix(x_red,[x_min,x_max],[y_min,y_max],'red')

def generatePlotData():
    x,t = sklearn.datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,
                              n_clusters_per_class=1, flip_y=-1,random_state=2)

    plotClassificationData(x,t)        
    for i in t:
        if t[i]==0:
            t[i]=-1
    return x,t

random.seed(1)

X,t=generatePlotData()

svm=SVM(X,t)
svm.fit()
svm.plotPredictionCurve()