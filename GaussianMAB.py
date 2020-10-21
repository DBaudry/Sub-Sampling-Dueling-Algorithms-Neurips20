""" Packages import """
from MAB import *


class GaussianMAB(GenericMAB):
    """
    Gaussian Bandit Problem
    """
    def __init__(self, p):
        """
        Initialization
        :param p: np.array, true values of 1/lambda for each arm
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['G']*len(p), p=p)
        # Parameters used for stop learning policy
        self.best_arm = self.get_best_arm()
        # Careful: Cp is the bound only with same variance for each arm
        self.Cp = sum([(self.mu_max - arm.mu) / self.kl2(arm.mu, self.mu_max, arm.eta, self.MAB[self.best_arm].eta)
                       for arm in self.MAB if arm.mu != self.mu_max])

    def get_best_arm(self):
        ind = np.nonzero(self.means == np.amax(self.means))[0]
        std = [self.MAB[arm].eta for arm in ind]
        u = np.argmin(std)
        return ind[u]

    @staticmethod
    def kl(mu1, mu2):
        """
        Implementation of the Kullback-Leibler divergence for two Gaussian N(mu, 1)
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return (mu2-mu1)**2/2

    @staticmethod
    def kl2(mu1, mu2, sigma1, sigma2):
        """
        Implementation of the Kullback-Leibler divergence for two Gaussian with different std
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return np.log(sigma2/sigma1) + 0.5 * (sigma1**2/sigma2**2 + (mu2-mu1)**2/sigma2**2 - 1)

    def TS(self, T):
        """
        Thompson Sampling for Gaussian distributions with known variance, and an inproper uniform prior
        on the mean
        :param T: Time Horizon
        :return: Tracker2 object
        """
        eta = np.array([arm.eta for arm in self.MAB])

        def f(x):
            return np.random.normal(x.Sa/x.Na, eta/np.sqrt(x.Na))
        return self.Index_Policy(T, f)

    def kl_ucb(self, T, f):
        """
        Implementation of KL-UCB for Gaussian bandits
        :param T: Time Horizon
        :param rho: coefficient for the upper bound
        :return:
        """
        def index_func(x):
            return x.Sa / x.Na + np.sqrt(f(x.t)*2 / x.Na)
        return self.Index_Policy(T, index_func)
