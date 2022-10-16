import numpy as np
import matplotlib.pyplot as plt

class RBF(object):

    def __init__(self, params):
        self.params = params
        self.ndim = len(params) - 1
    
    def kernMat(self,x1,x2):
        k_matrix = np.empty([np.shape(x1)[0],np.shape(x2)[0]])
        for i in range(0,np.shape(x1)[0]):
            for j in range(0,np.shape(x2)[0]):
                k_matrix[i][j] = self.computeKernel(x1[i,:],x2[j,:])
                #print(k_matrix[i][j])
        return k_matrix
    
    def computeKernel(self, x, y):
        return np.exp(-0.5 *7.0*(np.linalg.norm(x-y)**2) )
    
    def __call__(self, x, y):
        x, y = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        return self.params[0] * np.exp(-0.5 * np.sum(d, axis=-1))
   
    def _pairwise(self, x, y):
        return (
            np.tile(x, (len(y), 1, 1)).transpose(1, 0, 2),
            np.tile(y, (len(x), 1, 1))
        )
    
class RelevanceVectorRegressor:

    def __init__(self, kernel, alpha=1., beta=1.):
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, t, iter_max=10000):
        X = X[:, None]
        N = len(t)
        Phi = self.kernel.kernMat(X, X)
        Phi_init=np.copy(Phi)
        self.alpha = np.zeros(N) + self.alpha
        for _ in range(iter_max):
            params = np.hstack([self.alpha, self.beta])
            precision = np.diag(self.alpha) + self.beta * Phi.T .dot( Phi)
            covariance = np.linalg.inv(precision)
            mean = self.beta * covariance .dot( Phi.T ).dot( t)
            gamma = 1 - self.alpha * np.diag(covariance)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 1e10, out=self.alpha)
            self.beta = (N - np.sum(gamma)) / np.sum((t - Phi.dot(mean)) ** 2)
            if np.allclose(params, np.hstack([self.alpha, self.beta])):
                break
        mask = self.alpha < 1e9
        self.X = X[mask]
        self.t = t[mask]
        self.alpha = self.alpha[mask]
        Phi = self.kernel.kernMat(self.X, self.X)
        precision = np.diag(self.alpha) + self.beta * Phi.T .dot( Phi)
        self.covariance = np.linalg.inv(precision)
        self.mean = self.beta * self.covariance .dot( Phi.T) .dot( self.t)
        return Phi,Phi_init

    def predict(self, X, with_error=True):
        X = X[:, None]
        phi = self.kernel.kernMat(X, self.X)
        mean = phi .dot( self.mean)
        if with_error:
            var = 1 / self.beta + np.sum(phi .dot( self.covariance) * phi, axis=1)
            return phi,mean, np.sqrt(var)
        return phi,mean
    
def create_toy_data(n=10):
    x = np.linspace(0, 1, n)
    t = np.sin(2 * np.pi * x) + np.random.normal(scale=0.1, size=n)
    return x, t

x_train, y_train = create_toy_data(n=10)
x = np.linspace(0, 1, 100)

model = RelevanceVectorRegressor(RBF(np.array([1., 20.])))
Phi_aral,Phi_init_aral=model.fit(x_train, y_train)

phi_aral_pred,y, y_std = model.predict(x)

plt.scatter(x_train, y_train, facecolor="none", edgecolor="g", label="training")
plt.scatter(model.X.ravel(), model.t, s=100, facecolor="none", edgecolor="b", label="relevance vector")
plt.plot(x, y, color="r", label="predict mean")
plt.fill_between(x, y - y_std, y + y_std, color="pink", alpha=0.2, label="predict std.")
plt.legend(loc="best")
plt.show()