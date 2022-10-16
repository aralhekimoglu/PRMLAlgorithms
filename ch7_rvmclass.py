import numpy as np
import matplotlib.pyplot as plt

class Kernel(object):
    """
    Base class for kernel function
    """

    def _pairwise(self, x, y):
        """
        all pairs of x and y

        Parameters
        ----------
        x : (sample_size, n_features)
            input
        y : (sample_size, n_features)
            another input

        Returns
        -------
        output : tuple
            two array with shape (sample_size, sample_size, n_features)
        """
        return (
            np.tile(x, (len(y), 1, 1)).transpose(1, 0, 2),
            np.tile(y, (len(x), 1, 1))
        )

class RBF(Kernel):

    def __init__(self, params):
        """
        construct Radial basis kernel function

        Parameters
        ----------
        params : (ndim + 1,) ndarray
            parameters of radial basis function

        Attributes
        ----------
        ndim : int
            dimension of expected input data
        """
        assert params.ndim == 1
        self.params = params
        self.ndim = len(params) - 1

    def __call__(self, x, y, pairwise=True):
        """
        calculate radial basis function
        k(x, y) = c0 * exp(-0.5 * c1 * (x1 - y1) ** 2 ...)

        Parameters
        ----------
        x : ndarray [..., ndim]
            input of this kernel function
        y : ndarray [..., ndim]
            another input

        Returns
        -------
        output : ndarray
            output of this radial basis function
        """
        assert x.shape[-1] == self.ndim
        assert y.shape[-1] == self.ndim
        if pairwise:
            x, y = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        return self.params[0] * np.exp(-0.5 * np.sum(d, axis=-1))

    def derivatives(self, x, y, pairwise=True):
        if pairwise:
            x, y = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        delta = np.exp(-0.5 * np.sum(d, axis=-1))
        deltas = -0.5 * (x - y) ** 2 * (delta * self.params[0])[:, :, None]
        return np.concatenate((np.expand_dims(delta, 0), deltas.T))

    def update_parameters(self, updates):
        self.params += updates


class RelevanceVectorClassifier(object):

    def __init__(self, kernel, alpha=1.):
        """
        construct relevance vector classifier

        Parameters
        ----------
        kernel : Kernel
            kernel function to compute components of feature vectors
        alpha : float
            initial precision of prior weight distribution
        """
        self.kernel = kernel
        self.alpha = alpha

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _map_estimate(self, X, t, w, n_iter=10):
        for _ in range(n_iter):
            y = self._sigmoid(X .dot( w))
            g = X.T .dot( (y - t) )+ self.alpha * w
            H = (X.T * y * (1 - y)) .dot( X) + np.diag(self.alpha)
            w -= np.linalg.solve(H, g)
        return w, np.linalg.inv(H)

    def fit(self, X, t, iter_max=100):
        """
        maximize evidence with respect ot hyperparameter

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input
        t : (sample_size,) ndarray
            corresponding target
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        X : (N, n_features) ndarray
            relevance vector
        t : (N,) ndarray
            corresponding target
        alpha : (N,) ndarray
            hyperparameter for each weight or training sample
        cov : (N, N) ndarray
            covariance matrix of weight
        mean : (N,) ndarray
            mean of each weight
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        assert t.ndim == 1
        Phi = self.kernel(X, X)
        N = len(t)
        self.alpha = np.zeros(N) + self.alpha
        mean = np.zeros(N)
        for _ in range(iter_max):
            param = np.copy(self.alpha)
            mean, cov = self._map_estimate(Phi, t, mean, 10)
            gamma = 1 - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 1e10, out=self.alpha)
            if np.allclose(param, self.alpha):
                break
        mask = self.alpha < 1e8
        self.X = X[mask]
        self.t = t[mask]
        self.alpha = self.alpha[mask]
        Phi = self.kernel(self.X, self.X)
        mean = mean[mask]
        self.mean, self.covariance = self._map_estimate(Phi, self.t, mean, 100)

    def predict(self, X):
        """
        predict class label

        Parameters
        ----------
        X : (sample_size, n_features)
            input

        Returns
        -------
        label : (sample_size,) ndarray
            predicted label
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        label = (phi .dot( self.mean) > 0).astype(np.int)
        return label

    def predict_proba(self, X):
        """
        probability of input belonging class one

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input

        Returns
        -------
        proba : (sample_size,) ndarray
            probability of predictive distribution p(C1|x)
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        mu_a = phi .dot(self.mean)
        var_a = np.sum(phi .dot( self.covariance) * phi, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
    
def create_toy_data():
    x0 = np.random.normal(size=100).reshape(-1, 2) - 1.
    x1 = np.random.normal(size=100).reshape(-1, 2) + 1.
    x = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int)
    return x, y

x_train, y_train = create_toy_data()

model = RelevanceVectorClassifier(RBF(np.array([1., 0.5, 0.5])))
model.fit(x_train, y_train)

x0, x1 = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
x = np.array([x0, x1]).reshape(2, -1).T
plt.scatter(x_train[:, 0], x_train[:, 1], s=40, c=y_train, marker="x")
plt.scatter(model.X[:, 0], model.X[:, 1], s=100, facecolor="none", edgecolor="g")
plt.contourf(x0, x1, model.predict_proba(x).reshape(100, 100), np.linspace(0, 1, 5), alpha=0.2)
plt.colorbar()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect("equal", adjustable="box")