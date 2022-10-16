import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import gamma

np.random.seed(1234)
np.seterr(all="ignore")

def metropolis(func, rv, n, downsample=1):
    """
    Metropolis algorithm

    Parameters
    ----------
    func : callable
        (un)normalized distribution to be sampled from
    rv : RandomVariable
        proposal distribution which is symmetric at the origin
    n : int
        number of samples to draw
    downsample : int
        downsampling factor

    Returns
    -------
    sample : (n, ndim) ndarray
        generated sample
    """
    x = np.zeros((1, rv.ndim))
    sample = []
    for i in range(n * downsample):
        x_new = x + rv.draw()
        accept_proba = func(x_new) / func(x)
        if random.random() < accept_proba:
            x = x_new
        if i % downsample == 0:
            sample.append(x[0])
    sample = np.asarray(sample)
    assert sample.shape == (n, rv.ndim), sample.shape
    return sample



class RandomVariable(object):
    """
    base class for random variables
    """

    def __init__(self):
        self.parameter = {}

    def __repr__(self):
        string = "{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * 4)
            if isinstance(value, RandomVariable):
                string += "{key}={value:8}"
            else:
                string += "{key}={value}"
            string += "\n"
        string += ")"
        return string

    def __format__(self, indent="4"):
        indent = int(indent)
        string = "{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * indent)
            if isinstance(value, RandomVariable):
                string += "{key}=" + value.__format__(str(indent + 4))
            else:
                string += "{key}={value}"
            string += "\n"
        string += (" " * (indent - 4)) + ")"
        return string

    def fit(self, X, **kwargs):
        """
        estimate parameter(s) of the distribution

        Parameters
        ----------
        X : np.ndarray
            observed data
        """
        self._check_input(X)
        if hasattr(self, "_fit"):
            self._fit(X, **kwargs)
        else:
            raise NotImplementedError

    def pdf(self, X):
        """
        compute probability density function
        p(X|parameter)

        Parameters
        ----------
        X : (sample_size, ndim) np.ndarray
            input of the function

        Returns
        -------
        p : (sample_size,) np.ndarray
            value of probability density function for each input
        """
        self._check_input(X)
        if hasattr(self, "_pdf"):
            return self._pdf(X)
        else:
            raise NotImplementedError

    def draw(self, sample_size=1):
        """
        draw samples from the distribution

        Parameters
        ----------
        sample_size : int
            sample size

        Returns
        -------
        sample : (sample_size, ndim) np.ndarray
            generated samples from the distribution
        """
        assert isinstance(sample_size, int)
        if hasattr(self, "_draw"):
            return self._draw(sample_size)
        else:
            raise NotImplementedError

    def _check_input(self, X):
        assert isinstance(X, np.ndarray)



class Gamma(RandomVariable):
    """
    Gamma distribution
    p(x|a, b)
    = b^a x^(a-1) exp(-bx) / gamma(a)
    """

    def __init__(self, a, b):
        """
        construct Gamma distribution

        Parameters
        ----------
        a : int, float, or np.ndarray
            shape parameter
        b : int, float, or np.ndarray
            rate parameter
        """
        super().__init__()
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.shape == b.shape
        self.a = a
        self.b = b

    @property
    def a(self):
        return self.parameter["a"]

    @a.setter
    def a(self, a):
        if isinstance(a, (int, float, np.number)):
            if a <= 0:
                raise ValueError("a must be positive")
            self.parameter["a"] = np.asarray(a)
        elif isinstance(a, np.ndarray):
            if (a <= 0).any():
                raise ValueError("a must be positive")
            self.parameter["a"] = a
        else:
            if a is not None:
                print "error"
            self.parameter["a"] = None

    @property
    def b(self):
        return self.parameter["b"]

    @b.setter
    def b(self, b):
        if isinstance(b, (int, float, np.number)):
            if b <= 0:
                raise ValueError("b must be positive")
            self.parameter["b"] = np.asarray(b)
        elif isinstance(b, np.ndarray):
            if (b <= 0).any():
                raise ValueError("b must be positive")
            self.parameter["b"] = b
        else:
            if b is not None:
                print "error"
            self.parameter["b"] = None

    @property
    def ndim(self):
        return self.a.ndim

    @property
    def shape(self):
        return self.a.shape

    @property
    def size(self):
        return self.a.size

    def _pdf(self, X):
        return (
            self.b ** self.a
            * X ** (self.a - 1)
            * np.exp(-self.b * X)
            / gamma(self.a))

    def _draw(self, sample_size=1):
        return np.random.gamma(
            shape=self.a,
            scale=1 / self.b,
            size=(sample_size,) + self.shape
        )


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu, var)
    = exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
    """

    def __init__(self, mu=None, var=None, tau=None):
        super(Gaussian,self).__init__()
        self.mu = mu
        if var is not None:
            self.var = var
        elif tau is not None:
            self.tau = tau
        else:
            self.var = None
            self.tau = None

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            self.parameter["mu"] = np.array(mu)
        elif isinstance(mu, np.ndarray):
            self.parameter["mu"] = mu
        elif isinstance(mu, Gaussian):
            self.parameter["mu"] = mu
        else:
            if mu is not None:
                print "hata"
            self.parameter["mu"] = None

    @property
    def var(self):
        return self.parameter["var"]

    @var.setter
    def var(self, var):
        if isinstance(var, (int, float, np.number)):
            assert var > 0
            var = np.array(var)
            assert var.shape == self.shape
            self.parameter["var"] = var
            self.parameter["tau"] = 1 / var
        elif isinstance(var, np.ndarray):
            assert (var > 0).all()
            assert var.shape == self.shape
            self.parameter["var"] = var
            self.parameter["tau"] = 1 / var
        else:
            assert var is None
            self.parameter["var"] = None
            self.parameter["tau"] = None

    @property
    def tau(self):
        return self.parameter["tau"]

    @tau.setter
    def tau(self, tau):
        if isinstance(tau, (int, float, np.number)):
            assert tau > 0
            tau = np.array(tau)
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = 1 / tau
        elif isinstance(tau, np.ndarray):
            assert (tau > 0).all()
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = 1 / tau
        elif isinstance(tau, Gamma):
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = None
        else:
            assert tau is None
            self.parameter["tau"] = None
            self.parameter["var"] = None

    @property
    def ndim(self):
        if hasattr(self.mu, "ndim"):
            return self.mu.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.mu, "size"):
            return self.mu.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.mu, "shape"):
            return self.mu.shape
        else:
            return None

    def _fit(self, X):
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        if mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError
        elif mu_is_gaussian:
            self._bayes_mu(X)
        elif tau_is_gamma:
            self._bayes_tau(X)
        else:
            self._ml(X)

    def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

    def _map(self, X):
        assert isinstance(self.mu, Gaussian)
        assert isinstance(self.var, np.ndarray)
        N = len(X)
        mu = np.mean(X, 0)
        self.mu = (
            (self.tau * self.mu.mu + N * self.mu.tau * mu)
            / (N * self.mu.tau + self.tau)
        )

    def _bayes_mu(self, X):
        N = len(X)
        mu = np.mean(X, 0)
        tau = self.mu.tau + N * self.tau
        self.mu = Gaussian(
            mu=(self.mu.mu * self.mu.tau + N * mu * self.tau) / tau,
            tau=tau
        )

    def _bayes_tau(self, X):
        N = len(X)
        var = np.var(X, axis=0)
        a = self.tau.a + 0.5 * N
        b = self.tau.b + 0.5 * N * var
        self.tau = Gamma(a, b)

    def _bayes(self, X):
        N = len(X)
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        if mu_is_gaussian and not tau_is_gamma:
            mu = np.mean(X, 0)
            tau = self.mu.tau + N * self.tau
            self.mu = Gaussian(
                mu=(self.mu.mu * self.mu.tau + N * mu * self.tau) / tau,
                tau=tau
            )
        elif not mu_is_gaussian and tau_is_gamma:
            var = np.var(X, axis=0)
            a = self.tau.a + 0.5 * N
            b = self.tau.b + 0.5 * N * var
            self.tau = Gamma(a, b)
        elif mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _pdf(self, X):
        d = X - self.mu
        return (
            np.exp(-0.5 * self.tau * d ** 2) / np.sqrt(2 * np.pi * self.var)
        )

    def _draw(self, sample_size=1):
        return np.random.normal(
            loc=self.mu,
            scale=np.sqrt(self.var),
            size=(sample_size,) + self.shape
        )
        
def func(x):
    return np.exp(-x ** 2) + 3 * np.exp(-(x - 3) ** 2)
x = np.linspace(-5, 10, 100)
rv = Gaussian(mu=np.array([2.]), var=np.array([2.]))
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.plot(x, 15 * rv.pdf(x), label=r"$kq(z)$")
plt.fill_between(x, func(x), 15 * rv.pdf(x), color="gray")
plt.legend(fontsize=15)
plt.show()

samples = metropolis(func, Gaussian(mu=np.zeros(1), var=np.ones(1)), n=100, downsample=10)
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.hist(samples, normed=True, alpha=0.2)
plt.scatter(samples, np.random.normal(scale=.03, size=(100, 1)), s=5, label="samples")
plt.legend(fontsize=15)
plt.show()