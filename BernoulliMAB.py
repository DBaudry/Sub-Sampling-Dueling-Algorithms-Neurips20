""" Packages import """
from MAB import *
from utils import  rollavg_bottlneck, rd_choice, hypergeom_sample
from scipy.optimize import brentq


class BetaBernoulliMAB(GenericMAB):
    """
    Bernoulli Bandit Problem
    """
    def __init__(self, p):
        """
        Initialization
        :param p: np.array, true probabilities of success for each arm
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['B']*len(p), p=p)
        # Complexity
        self.Cp = sum([(self.mu_max-x)/self.kl(x, self.mu_max) for x in self.means if x != self.mu_max])

    @staticmethod
    def kl(x, y):
        """
        Implementation of the Kullback-Leibler divergence for two Bernoulli distributions (B(x),B(y))
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        if x == y:
            return 0
        elif x > 1-1e-6:
            return 0
        elif y == 0 or y == 1:
            return np.inf
        elif x < 1e-6:
            return (1-x) * np.log((1-x)/(1-y))
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

    def TS(self, T):
        """
        Beta-Bernoulli Thompson Sampling
        :param T: Time Horizon
        :return: Tracker2 object
        """
        def f(x):
            return np.random.beta(x.Sa+1, x.Na-x.Sa+1)
        return self.Index_Policy(T, f)

    def BESA_duel(self, indices, tracker):
        """
        More efficient implementation of the BESA duel in the Bernoulli case
        :param indices: indices of arms of the duel
        :param tracker: Tracker2 object
        :return: winner of the duel
        """
        i, j = indices[0], indices[1]
        ni, nj = tracker.Na[i], tracker.Na[j]
        si, sj = tracker.Sa[i], tracker.Sa[j]
        idx_min = np.argmin([ni, nj])
        if idx_min == 0:
            sj = hypergeom_sample(sj, nj, ni)
        else:
            si = hypergeom_sample(si, ni, nj)
        return indices[rd_argmax(np.array([si, sj]))]

    def SSMC(self, T, explo_func=lambda x: np.sqrt(np.log(x))):
        """
        More efficient implementation of SSMC for the Bernoulli case
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker2 object
        """
        tr = Tracker2(self.means, T, store_rewards_arm=True)
        r, t, l = 1, 0, -1
        while t < self.nb_arms:
            arm = t
            tr.update(t, arm, self.MAB[arm].sample()[0])
            t += 1
        while t < T:
            l_prev = l
            l = get_leader(tr.Na, tr.Sa, l_prev)
            t_prev, forced_explo = t, explo_func(r)

            indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            if l_prev != l or tr.rewards_arm[l][-1] == 0:
                for j in range(self.nb_arms):
                    if indic[j] == 0 and j != l:
                        if l_prev == l:
                            lead_min = np.mean(tr.rewards_arm[l][-int(tr.Na[j]):])
                        else:
                            lead_min = rollavg_bottlneck(tr.rewards_arm[l], int(tr.Na[j]))[(int(tr.Na[j])-1):].min()
                        if tr.Sa[j]/tr.Na[j] >= lead_min and t < T:
                            indic[j] = 1
            if indic.sum() == 0:
                tr.update(t, l, self.MAB[l].sample()[0])
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        tr.update(t, i, self.MAB[i].sample()[0])
                        t += 1
            r += 1
        return tr

    def PHE(self, T, a, distrib=None):
        """
        More efficient version of PHE for Bernoulli bandits
        :param T: Time Horizon
        :param a: proportion of perturbed history. a=1 -> same proportion, a=0-> no perturbed history
        :param distrib: distribution of the perturbed history
        :return: Tracker2 object
        """
        tr = Tracker2(self.means, T, store_rewards_arm=True)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                idx_mean = np.zeros(self.nb_arms)
                for k in range(self.nb_arms):
                    ph = np.random.binomial(n=np.int(a*tr.Na[k])+1, p=0.5)
                    idx_mean[k] = (tr.Sa[k]+ph)/(tr.Na[k]+np.int(a*tr.Na[k])+1)
                arm = rd_argmax(idx_mean)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def kl_ucb(self, T, f):
        """
        Implementation of the KL-UCB algorithm for Bernoulli bandits
        :param T: Time Horizon
        :param f: Function in the minimization problem
        :return: Tracker2 object
        """
        def index_func(x):
            res = []
            for k in range(self.nb_arms):
                if x.Sa[k]/x.Na[k] < 1e-6:
                    res.append(1)
                elif x.Sa[k]/x.Na[k] > 1-1e-6:
                    res.append(1)
                else:
                    def kl_shift(y):
                        return self.kl(x.Sa[k]/x.Na[k], y) - f(x.t)/x.Na[k]
                    res.append(brentq(kl_shift, x.Sa[k]/x.Na[k]-1e-7, 1 - 1e-10))
            return np.array(res)
        return self.Index_Policy(T, index_func)
