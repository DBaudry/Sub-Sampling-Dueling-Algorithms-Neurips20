""" Packages import """
from MAB import *
from scipy.optimize import brentq


class ExponentialMAB(GenericMAB):
    """
    Gaussian Bandit Problem
    """
    def __init__(self, p):
        """
        Initialization
        :param p: np.array, true values of (mu, sigma) for each arm with mean sampled from N(mu, sigma)
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['Exp']*len(p), p=p)
        # Parameters used for stop learning policy
        self.Cp = sum([(self.mu_max-x)/self.kl(1/x, 1/self.mu_max) for x in self.means if x != self.mu_max])


    @staticmethod
    def kl(x, y):
        """
        Implementation of the Kullback-Leibler divergence for two Exponential Distributions
        WARNING: x, y are the inverse of the means of the distributions
        :param x: float
        :param y: float
        :return: float, KL(E(x), E(y))
        """
        return np.log(x/y) + y/x - 1

    def TS(self, T):
        """
        Thompson Sampling with known variance, and an inproper uniform prior
         on the mean
        :param T: Time Horizon
        :return: Tracker2 object
        """
        def f(x):
            return 1/np.random.gamma(shape=x.Na, scale=1/x.Sa)
        return self.Index_Policy(T, f)

    def kl_ucb(self, T, f):
        """
        Implementation of KL-UCB for Exponential distributions
        :param T: Time Horizon
        :param f: function in the minimization problem
        :return: Tracker2 object
        """
        def index_func(x):
            res = []
            for k in range(self.nb_arms):
                mu = x.Sa[k] / x.Na[k]
                def kl_shift(y):
                    return np.log(y/mu) + mu/y-1 - f(x.t) / x.Na[k]
                res.append(brentq(kl_shift, mu*np.exp(f(x.t)/x.Na[k]), mu*np.exp(f(x.t)/x.Na[k]+1)))
            return np.array(res)

        return self.Index_Policy(T, index_func)

    def IMED(self, T):
        """
        Implementation of IMED for Exponential distributions
        :param T: Time Horizon
        :return: Tracker2 object
        """
        def index_func(x):
            mu_max = np.max(x.Sa/x.Na)
            idx = []
            for k in range(self.nb_arms):
                idx.append(x.Na[k]*self.kl(mu_max, x.Sa[k]/x.Na[k])+np.log(x.Na[k]))
            return -np.array(idx)
        return self.Index_Policy(T, index_func)
