import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(1234)

class RBF:

    def __init__(self,sigma):
        self.sigma = sigma
                       
    def pairwise(self, x, y):
        return (
            np.tile(x, (len(y), 1, 1)).transpose(1, 0, 2),
            np.tile(y, (len(x), 1, 1))
        )
    def computeKernel(self, x, y):
        return np.exp(-0.5 *7.0*(np.linalg.norm(x-y)**2) )
#    def computeKernel(self, x, y):
#        x, y = self.pairwise(x, y)
#        d = self.sigma * (x - y) ** 2
#        return np.exp(-0.5 * np.sum(d, axis=-1))
    def kernMat(self,x1,x2):
        k_matrix = np.empty([np.shape(x1)[0],np.shape(x2)[0]])
        for i in range(0,np.shape(x1)[0]):
            for j in range(0,np.shape(x2)[0]):
                k_matrix[i][j] = self.computeKernel(x1[i,:],x2[j,:])
                #print(k_matrix[i][j])
        return k_matrix
    
class GaussianProcessClassifier:

    def __init__(self, kernel, noise_level=1e-4):
        self.kernel = kernel
        self.noise_level = noise_level

    def sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X, t):
#        print X.ndim
#        if X.ndim == 1:
#            X = X[:, None]
        self.X = X
        self.t = t
        X_aral=X
        Gram=self.kernel.computeKernel(X, X)
        
        aral= Gram
        
        self.covariance = Gram + np.eye(Gram.size) * self.noise_level
        self.precision = np.linalg.inv(self.covariance)
        return X_aral,aral

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel.kernMat(X, self.X)
        print self.precision.shape
        a_mean = K .dot( self.precision).dot(  self.t)
        return K,self.sigmoid(a_mean)

def create_toy_data():
    x0 = np.random.normal(size=50).reshape(-1, 2)
    x1 = np.random.normal(size=50).reshape(-1, 2) + 2.
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)[:, None]


def pairwise(x):   
        return (
            np.tile(x, (len(x), 1, 1)).transpose(1, 0, 2),
            np.tile(x, (len(x), 1, 1))
        )

def kernel(x1,x2):
    return np.exp(-0.5 *7.0*(np.linalg.norm(x1-x2)**2) )

x_train, y_train = create_toy_data()
x0, x1 = np.meshgrid(np.linspace(-4, 6, 100), np.linspace(-4, 6, 100))
x = np.array([x0, x1]).reshape(2, -1).T

model = GaussianProcessClassifier( RBF(sigma=7.) )
X_aral,aral=model.fit(x_train, y_train)
x_aral, y_aral = pairwise(X_aral)
K_predict_aral,y = model.predict(x)

#kernel_matrixx=kernel_matrix(X_aral,X_aral)


plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x0, x1, y.reshape(100, 100), levels=np.linspace(0,1,3), alpha=0.2)
plt.xlim(-4, 6)
plt.ylim(-4, 6)