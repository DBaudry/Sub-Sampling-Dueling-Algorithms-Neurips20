""" Packages import """
from MAB import *
from scipy.stats import norm
from tracker import Tracker2
import numpy as np


class TruncGaussianMAB(GenericMAB):
    """
    Gaussian Bandit Problem
    """
    def __init__(self, p):
        """
        Initialization
        :param p: np.array, true values of 1/lambda for each arm
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['TG']*len(p), p=p)
        self.best_arm = self.get_best_arm()
        self.Cp = self.get_complexity()

    def get_complexity(self):
        """
        :return: Compute the constant in the Burnetas and Katehakis lower bound for TG arms
        """
        Cp = 0
        for arm in self.MAB:
            if arm.mean != self.mu_max:
                gap = self.mu_max-arm.mean
                kl = self.KL_tg(arm.mu, self.MAB[self.best_arm].mu, arm.scale)
                Cp += gap/kl
        return Cp

    @staticmethod
    def KL_tg(mu1, mu2, scale, step=1e-6):
        """
        :param mu1: mean of underlying Gaussian r.v of arm 1
        :param mu2: mean of underlying Gaussian r.v of arm 1
        :param scale: scale of underlying Gaussian r.v
        :param step: precision of numerical integration
        :return: KL divergence of two TG arms
        """
        phi01 = norm.cdf(0, loc=mu1, scale=scale)
        phi02 = norm.cdf(0, loc=mu2, scale=scale)
        phi11 = 1 - norm.cdf(1, loc=mu1, scale=scale)
        phi12 = 1 - norm.cdf(1, loc=mu2, scale=scale)
        kl_1 = phi01 * np.log(phi01 / phi02) + phi11 * np.log(phi11 / phi12)
        X = np.arange(0, 1, step)
        kl_2 = (norm.pdf(X, loc=mu1, scale=scale) * np.log(norm.pdf(
            X, loc=mu1, scale=scale) / norm.pdf(X, loc=mu2, scale=scale))).mean()
        return kl_1 + kl_2

    def get_best_arm(self):
        """
        :return: best arm of the bandit problem
        """
        ind = np.nonzero(self.means == np.amax(self.means))[0]
        std = [self.MAB[arm].scale for arm in ind]
        u = np.argmin(std)
        return ind[u]

    def PHE(self, T, a, distrib=None):
        """
        Optimized version of PHE for TG arms
        :param T: Time Horizon
        :param a: proportion of perturbed history. a=1 -> same proportion, a=0-> no perturbed history
        :param distrib: distribution of the perturbed history
        :return:
        """
        tr = Tracker2(self.means, T, store_rewards_arm=True)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                idx_mean = np.zeros(self.nb_arms)
                for k in range(self.nb_arms):
                    ph = np.random.binomial(n=np.int(a*tr.Na[k]), p=0.5)
                    idx_mean[k] = (tr.Sa[k]+ph)/(tr.Na[k]+np.int(a*tr.Na[k]))
                arm = rd_argmax(idx_mean)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def TS(self, T):
        """
        Implementation of Thompson Sampling with a Binarization trick
        :param T: Time Horizon
        :return: Tracker2 object
        """
        def f(S, N):
            return np.random.beta(S+1, N-S+1)
        tr = Tracker2(self.means, T)
        bin_Sa = np.zeros(self.nb_arms)
        for t in range(T):
            if t < self.nb_arms:
                arm = t % self.nb_arms
            else:
                arm = rd_argmax(f(bin_Sa, tr.Na))
            reward = self.MAB[arm].sample()[0]
            bin_Sa[arm] += np.random.binomial(n=1, p=reward)
            tr.update(t, arm, reward)
        return tr

    def IMED(self, T):
        """
        Implementation of IMED with a binarization trick
        :param T:
        :return:
        """
        def kl_ber(x, y):
            if x == y:
                return 0
            elif x > 1 - 1e-6:
                return 0
            elif y == 0 or y == 1:
                return np.inf
            elif x < 1e-6:
                return (1 - x) * np.log((1 - x) / (1 - y))
            return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

        def index_func(bin_Sa, x):
            mu_max = np.max(bin_Sa/x.Na)
            idx = []
            for k in range(self.nb_arms):
                idx.append(x.Na[k]*kl_ber(bin_Sa[k]/x.Na[k], mu_max)+np.log(x.Na[k]))
            return -np.array(idx)
        tr = Tracker2(self.means, T)
        bin_Sa = np.zeros(self.nb_arms)
        for t in range(T):
            if t < self.nb_arms:
                arm = t % self.nb_arms
            else:
                arm = rd_argmax(index_func(bin_Sa, tr))
            reward = self.MAB[arm].sample()[0]
            bin_Sa[arm] += np.random.binomial(n=1, p=reward)
            tr.update(t, arm, reward)
        return tr