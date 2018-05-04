import matplotlib.pyplot as plt
import numpy as np
from math import pi as pi

class BaseRegression:
    def __init__(self):
        self._learned = False
        self._Sigma_N = np.NaN
        self._mean_N = np.NaN
    
    @property
    def learned(self):
        return self._learned
    
    @property
    def Sigma_N(self):
        return self._Sigma_N
    
    @property
    def mean_N(self):
        return self._mean_N

    @learned.setter
    def learned(self, value):
        self._learned = value
    
    @Sigma_N.setter
    def Sigma_N(self, value):
        self._Sigma_N = value
    
    @mean_N.setter
    def mean_N(self, value):
        self._mean_N = value
        
class BayesianRegression(BaseRegression):
    def fit(self, x, y, Sigma_0, beta=0):
        """
        Args:
            x (np.array): Training data of shape[n_samples, poly_degree]
            y (np.array): Target values of shape[n_samples, 1]
            lamb (float): regularization parameter lambda
            
        Returns: an instance of self
        """
        #Calculate Posterior, again as Gaussian
        Sigma_0_inv=np.linalg.inv(Sigma_0)         
        self.Sigma_N=np.linalg.inv(Sigma_0_inv+beta*x.T.dot(x))
        self.mean_N=beta*Sigma_N.dot(x.T).dot(y) # mean of posterior also w in wTx for regression
        self.learned=True
        return self
    
    def predict(self, x):
        """
        Args:
            x (np.array): Test data of shape[1, 1]

        Returns:
            prediction (np.array): shape[1, 1], predicted values

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Model has not been fitted, fit first before prediction')
        
        return np.dot(self.mean_N, gaussian_basis_function(x))
    
    def plot_prediction(self,x_range=100):
        """
        Args:
            x_range (integer): x_range of data to be plotted

        Returns:
            an instance of self

        Raises:
            ValueError if model has not been fit
        """
        x_pred = np.arange(-2*pi, 2*pi, pi/1000.0)
        y_pred = [self.predict(i) for i in x_pred] 
        plt.plot(x_pred,y_pred,'r')

## generate sinusoidal data with noise option
def generate_sinusodial_data(amount=100,add_noise=True):
    x=np.arange(-2*pi,2*pi,4*pi/amount)
    y=[np.sin(i) for i in x]
    #add noise
    if(add_noise):
        x=[np.random.normal(i,0.03) for i in x]
        y=[np.random.normal(i,0.07) for i in y]
    return x,y


#gaussian basis function to have nonlinear feature vector
def gaussian_basis_function(x,sigma=pi,xmin=-2*pi,xmax=2*pi,numFeatures=20):
    return np.append(1,np.exp(-(x - np.arange(xmin,xmax, (xmax-xmin)/float(numFeatures))) ** 2 / (2 * sigma * sigma)))


#gaussian features
def gaussian_features(x,numFeatures=20):
    sigma = np.ptp(x)/4 
    xmax= np.amax(x)
    xmin= np.amin(x)
    return np.array([gaussian_basis_function(i,sigma,xmin,xmax,numFeatures) for i in x])

x,y= generate_sinusodial_data(100,add_noise=True)    
plt.scatter(x,y)
gaus_features=gaussian_features(x)                   

#Bayesion Regression
num_training_set=gaus_features.shape[0]
num_features=gaus_features.shape[1]
beta=100.0 # noise parameter

#Prior as Gaussian
#S0 can be any num_featuresxnum_features matrix
#m0 can be any num_featuresx1 matrix
alpha=0.001
Sigma_0=(1/float(alpha))*np.identity(num_features)
mean_0=np.zeros((num_features,1))

bayesionRegressor=BayesianRegression()
bayesionRegressor.fit(gaus_features,y,Sigma_0,beta)
bayesionRegressor.plot_prediction()
