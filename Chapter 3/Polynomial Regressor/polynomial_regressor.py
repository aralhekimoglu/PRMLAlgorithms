import random
import numpy as np
import math
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self):
        self._learned = False
        self._weights = np.NaN
        self._polynomialDegree = 1
    
    @property
    def learned(self):
        return self._learned
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def polynomialDegree(self):
        return self._polynomialDegree

    @learned.setter
    def learned(self, value):
        self._learned = value
    
    @weights.setter
    def weights(self, value):
        self._weights = value
    
    @polynomialDegree.setter
    def polynomialDegree(self, value):
        self._polynomialDegree = value

    def fit(self, x, y, lamb=0):
        """
        Args:
            x (np.array): Training data of shape[n_samples, poly_degree]
            y (np.array): Target values of shape[n_samples, 1]
            lamb (float): regularization parameter lambda
            
        Returns: an instance of self
        """
        # fit w to data
        x_t=x.transpose()
        xt_x=x_t.dot(x)
        self.weights=np.linalg.pinv(lamb*np.identity(xt_x.shape[1])+xt_x).dot(x_t).dot(y)
        self.learned=True
        self.polynomialDegree=self.weights.size-1
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
        x_poly=np.matrix(polynomial_features(np.matrix(x),degree=self.polynomialDegree)) 
        prediction=x_poly.dot(self.weights)  
        return prediction
        
    def plot_prediction(self,x_range=100):
        """
        Args:
            x_range (integer): x_range of data to be plotted

        Returns:
            an instance of self

        Raises:
            ValueError if model has not been fit
        """
        x_pred=np.matrix([i for i in range (0,x_range)]).transpose()
        x_pred_poly=np.matrix(polynomial_features(x_pred,degree=self.polynomialDegree)) 
        for i in range (1,self.weights.size):
            x_pred_poly[:,i]=(x_pred_poly[:,i]-mean_poly[i])/variance_poly[i]  
        y_pred=x_pred_poly.dot(self.weights)    
        plt.plot(x_pred,y_pred,color='red',label='yes')
        return self

def generate_polynomial_data(degree=1, amount=10, x_range=10):
    """
        Args:
            degree (int): Degree of polynomial
            amount (int): Amount of data to be generated
            x_range (int): Range of horizontal axis
            
        Returns: 
            x (np.array) of shape [amount,1]: Generated data for horizontal axis
            y (np.array) of shape [amount,1]: Generated data for vertical axis
    """
    y=[]
    rng=np.arange(0.0, x_range, x_range/float(amount))
    x=np.matrix([i for i in rng]).transpose()
    for i in rng:
        powered_i=math.pow(i,degree)
        y.append(powered_i+random.uniform(-powered_i*0.1,powered_i*0.1))
    y=np.matrix(y).transpose()
    return x,y

def feature_scaling(X):
    """
        Args:
            X (np.matrix) of shape [X.shape[0],X.shape[1]]: Matrix to be feature scaled
            
        Returns: 
            X (np.matrix) of shape [X.shape[0],X.shape[1]]: Feature scaling applied matrix
            mean_X (np.array) of shape [X.shape[1],1]: mean of each column of X
            var_X (np.array) of shape [X.shape[1],1]: variance of each column of X
    """
    X=X.astype(float)
    mean_X=[]
    var_X=[]
    col_size=X.shape[1]
    for i in range(0,col_size):
        mean_i=np.mean(X[:,i])
        mean_X.append(mean_i)
        var_i=np.var(X[:,i])
        var_X.append(var_i)
        if (var_i!=0):
            X[:,i]=(X[:,i]-mean_i)/float(var_i)
    return X,np.array(mean_X),np.array(var_X)

def polynomial_features(x,degree=2):
    """
        Args:
            x (np.array): regular features
            degree (int): polynomial degree to convert features
            
        Returns: 
            x_poly (np.array) of shape x.shape
        """
    row_size=x.size
    x_poly=np.zeros((row_size,degree+1))
    for i in range (0,row_size):
        for j in range (0, degree+1):
            x_poly[i,j]=math.pow(x[i,:],j)
    return x_poly

# these are parameters that can be played with
poly_degree=3
x_range=100
amountOfData=100
lamb=0.0

# generate polynomial data
x,y=generate_polynomial_data(degree=poly_degree,amount=amountOfData,x_range=x_range)

# compute polynomial features for x
x_poly=polynomial_features(x,poly_degree)
x_poly_scaled,mean_poly,variance_poly=feature_scaling(x_poly)

# plot generated data
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')

regressor=PolynomialRegression()
regressor.fit(x_poly_scaled,y,lamb)
regressor.plot_prediction(x_range)