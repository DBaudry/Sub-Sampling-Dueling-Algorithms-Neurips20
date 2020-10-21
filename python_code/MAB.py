""" Packages import """
import numpy as np
import arms
from tqdm import tqdm
from utils import rd_argmax, rd_choice, rollavg_bottlneck, get_leader
from tracker import Tracker2
from utils import get_SSMC_star_min
#import sobol_seq # for LDS-SDA

mapping = {'B': arms.ArmBernoulli, 'beta': arms.ArmBeta, 'F': arms.ArmFinite, 'G': arms.ArmGaussian,
           'Exp': arms.ArmExponential, 'dirac': arms.dirac, 'TG': arms.ArmTG}


def default_exp(x):
    """
    :param x: float 
    :return: default exploration function for SDA algorithms
    """
    return 0
    # return np.sqrt(np.log(x))


class GenericMAB:
    """
    Generic class to simulate a Multi-Arm Bandit problem
    """
    def __init__(self, methods, p):
        """
        Initialization of the arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        """
        self.MAB = self.generate_arms(methods, p)
        self.nb_arms = len(self.MAB)
        self.means = np.array([el.mean for el in self.MAB])
        self.mu_max = np.max(self.means)
        self.mc_regret = None

    @staticmethod
    def generate_arms(methods, p):
        """
        Method for generating different arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        :return: list of class objects, list of arms
        """
        arms_list = list()
        for i, m in enumerate(methods):
            args = [p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(p[i]) == list else args
            alg = mapping[m]
            arms_list.append(alg(*args))
        return arms_list

    @staticmethod
    def kl(x, y):
        return None

    def MC_regret(self, method, N, T, param_dic, store_step=-1):
        """
        Average Regret on a Number of Experiments
        :param method: string, method used (UCB, Thomson Sampling, etc..)
        :param N: int, number of independent experiments
        :param T: int, time horizon
        :param param_dic: dict, parameters for the different methods
        """
        mc_regret = np.zeros(T)
        store = store_step > 0
        if store:
            all_regret = np.zeros((np.arange(T)[::store_step].shape[0], N))
        alg = self.__getattribute__(method)
        for i in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            tr = alg(T, **param_dic)
            regret = tr.regret()
            mc_regret += regret
            if store:
                all_regret[:, i] = regret[::store_step]
        if store:
            return mc_regret / N, all_regret
        return mc_regret / N

    def DummyPolicy(self, T):
        """
        Implementation of a random policy consisting in randomly choosing one of the available arms. Only useful
        for checking that the behavior of the different policies is normal
        :param T:  int, time horizon
        :return: means, arm sequence
        """
        tr = Tracker2(self.means, T)
        tr.arm_sequence = np.random.randint(self.nb_arms, size=T)
        return tr

    def ExploreCommit(self, T, m):
        """
        Implementation of Explore-then-Commit algorithm
        :param T: int, time horizon
        :param m: int, number of rounds before choosing the best action
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        tr = Tracker2(self.means, T)
        for t in range(m * self.nb_arms):
            arm = t % self.nb_arms
            tr.update(t, arm, self.MAB[arm].sample()[0])
        arm = rd_argmax(tr.Sa / tr.Na)
        for t in range(m * self.nb_arms, T):
            tr.update(t, arm, self.MAB[arm].sample()[0])
        return tr

    def Index_Policy(self, T, index_func, start_explo=1, store_rewards_arm=False):
        """
        Implementation of UCB1 algorithm
        :param T: int, time horizon
        :param start_explo: number of time to explore each arm before comparing index
        :param index_func: function which computes the index with the tracker
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        tr = Tracker2(self.means, T, store_rewards_arm)
        for t in range(T):
            if t < self.nb_arms*start_explo:
                arm = t % self.nb_arms
            else:
                arm = rd_argmax(index_func(tr))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def UCB1(self, T, rho=1.):
        """
        :param T: Time Horizon
        :param rho: coefficient for the upper bound
        :return:
        """
        def index_func(x):
            return x.Sa / x.Na + rho * np.sqrt(np.log(x.t + 1)*2 / x.Na)
        return self.Index_Policy(T, index_func)

    def BESA_duel(self, indices, tracker):
        """
        :param indices: indices of the 2 competing arms
        :param tracker: Tracker2 object
        :return: winner arm of a single dual in BESA
        """
        i, j = indices[0], indices[1]
        r_i, r_j = tracker.rewards_arm[i], tracker.rewards_arm[j]
        ni, nj = tracker.Na[i], tracker.Na[j]
        idx_max = rd_argmax(np.array([ni, nj]))
        if idx_max == 1:
            r_j = rd_choice(np.array(r_j), size=int(ni))
        else:
            r_i = rd_choice(np.array(r_i), size=int(nj))
        return indices[rd_argmax(np.array([np.mean(r_i), np.mean(r_j)]))]

    def BESA_step(self, tracker):
        """
        :param tracker: Tracker2 object
        :return: Implementation of the tournament in BESA
        """
        indices = list(np.arange(self.nb_arms))
        while len(indices) > 1:
            np.random.shuffle(indices)  # Changement pour enlever le biais
            winners = []
            if len(indices) % 2 == 1:
                winners.append(indices[-1])
            for i in range(len(indices)//2):
                winners.append(self.BESA_duel((indices[2*i], indices[2*i+1]), tracker))
            indices = winners
        return indices[0]

    def BESA(self, T, n0=1):
        """
        Implementation of the BESA algorithm
        :param T: Time Horizon 
        :param n0: Number of time to pull each arm before starting the algorithm
        :return: Tracker object with the results of the run
        """
        tr = Tracker2(self.means, T, store_rewards_arm=True)
        for t in range(T):
            if t < self.nb_arms * n0:
                arm = t % self.nb_arms
            else:
                arm = self.BESA_step(tr)
            tr.update(t, arm, self.MAB[arm].sample()[0])
        return tr

    def SSMC(self, T, explo_func=lambda x: np.sqrt(np.log(x))):
        """
        Implementation of the SSMC algorithm
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
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
    
    def SSMC_star(self, T, explo_func=default_exp):
        """
        Implemention of SSMC*, a slightly modified version of SSMC
        :param T: Time Horizon 
        :param explo_func: Forced Exploration function
        :return: Tracker object with the results of the run
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
            all_reshape_size = np.zeros(self.nb_arms)
            indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            for j in range(self.nb_arms):
                reshape_size = len(tr.rewards_arm[l]) // tr.Na[j]
                if indic[j] == 0 and j != l:
                    if l_prev == l and reshape_size == all_reshape_size[j]:
                        lead_min = np.inf
                    elif l_prev == l:
                        lead_min = np.mean(tr.rewards_arm[l][-int(tr.Na[j]):])
                    else:
                        lead_min = get_SSMC_star_min(tr.rewards_arm[l],
                                                     int(tr.Na[j]), int(reshape_size))
                    if tr.Sa[j]/tr.Na[j] >= lead_min and t < T:
                        indic[j] = 1
                all_reshape_size[j] = reshape_size
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

    def non_parametric_TS(self, T, upper_bound=1):
        """
        Implementation of the Non-parametric Thompson Sampling algorithm
        :param T: Time Horizon
        :param upper_bound: Upper bound for the reward
        :return: Tracker object with the results of the run
        """
        tr = Tracker2(self.means, T)
        if upper_bound is not None:
            X = [[upper_bound] for _ in range(self.nb_arms)]
        tr.Na = tr.Na + 1
        for t in range(T):
            V = np.zeros(self.nb_arms)
            for i in range(self.nb_arms):
                V[i] = np.inner(np.random.dirichlet(np.ones(int(tr.Na[i]))), np.array(X[i]))
            arm = rd_argmax(V)
            tr.update(t, arm, self.MAB[arm].sample()[0])
            X[arm].append(tr.reward[t])
        return tr

    def WR_SDA(self, T, explo_func=default_exp):
        """
        Implementation of WR-SDA
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
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
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l:
                    if self.BESA_duel([l, j], tracker=tr) == j:
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

    def RB_SDA(self, T, explo_func=default_exp):
        """
        Implementation of RB-SDA
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
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
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    tj = np.random.randint(tr.Na[l]-tr.Na[j])
                    lead_mean = np.mean(tr.rewards_arm[l][tj: tj+int(tr.Na[j])])
                    if tr.Sa[j]/tr.Na[j] >= lead_mean and t < T:
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

    def IB_SDA(self, T, explo_func=default_exp):
        """
        Implementation of IB-SDA (Independent Blocks-SDA): an algorithm not introduced in the paper 
        using a SWR sampler which discards elements that were previously drawn until there are no
        more available elements. It is a way to enforce the diversity of sample.
        We did not present this sampler as it is not an independent sampler.
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
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
            if l_prev != l:
                weight_dic = np.ones((self.nb_arms, int(tr.Na[l])))
            else:
                if weight_dic.shape[1] < tr.Na[l]:
                    weight_dic = np.concatenate([weight_dic,
                                             np.ones((self.nb_arms, 1))], axis=1)
                for k in range(self.nb_arms):
                    if k != l and weight_dic[k].sum() < tr.Na[k]:
                        weight_dic[k] = np.ones(int(tr.Na[l]))
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    besa_indices = np.random.choice(
                        np.arange(tr.Na[l]).astype('int'), size=int(tr.Na[j]),
                        replace=False, p=weight_dic[j]/weight_dic[j].sum())
                    lead_mean = np.mean(np.array(tr.rewards_arm[l])[besa_indices])
                    weight_dic[j][besa_indices] = 0
                    if tr.Sa[j]/tr.Na[j] >= lead_mean and t < T:
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

    def LB_SDA(self, T, explo_func=default_exp):
        """
        Implementation of the LB-SDA algorithm
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
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
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    lead_mean = np.mean(tr.rewards_arm[l][-int(tr.Na[j]):])
                    if tr.Sa[j]/tr.Na[j] >= lead_mean and t < T:
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

    def LDS_SDA(self, T, explo_func=default_exp):
        """
        Implementation of the LDS-SDA algorithm using a Sobol sequence
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
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
            t_b = int(sobol_seq.i4_sobol(1, seed=r)[0][0]*tr.Na[l])
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    b_0 = tr.rewards_arm[l][t_b:t_b+int(tr.Na[j])]
                    if len(b_0) < tr.Na[j]:
                        b_1 = tr.rewards_arm[l][:int(tr.Na[j])-int(tr.Na[l]-t_b)]
                        lead_mean = (np.sum(b_0)+np.sum(b_1))/tr.Na[j]
                    else:
                        lead_mean = np.mean(b_0)
                    if tr.Sa[j]/tr.Na[j] >= lead_mean and t < T:
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

    def vanilla_bootstrap(self, T):
        """
        Implementation of the Vanilla Bootstrap bandit algorithm 
        :param T: Time Horizon
         :return: Tracker object with the results of the run
        """
        tr = Tracker2(self.means, T, store_rewards_arm=True)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                bts_mean = np.zeros(self.nb_arms)
                for k in range(self.nb_arms):
                    bts_mean[k] = np.random.choice(tr.rewards_arm[k], size=int(tr.Na[k]), replace=True).mean()
                arm = rd_argmax(bts_mean)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def PHE(self, T, a, distrib):
        """
        Implementation of the Perturbed History Exploration algorithm
        :param T: Time Horizon
        :param a: proportion of perturbed history. a=1 -> same proportion, a=0-> no perturbed history
        :param distrib: Distribution of the fake rewards
        :return: Tracker2 object
        """
        tr = Tracker2(self.means, T, store_rewards_arm=True)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                idx_mean = np.zeros(self.nb_arms)
                for k in range(self.nb_arms):
                    ph = distrib.rvs(size=np.int(a*tr.Na[k])+1)
                    idx_mean[k] = (tr.Sa[k]+ph.sum())/(tr.Na[k]+np.int(a*tr.Na[k])+1)
                arm = rd_argmax(idx_mean)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def ReBoot(self, T, sigma, weight_func=np.random.normal):
        """
        Implementation of the Reboot algorithm
        :param T: Time Horizon 
        :param sigma: sigma and -sigma are added to the rewards list before bootstrapping
        :param weight_func: a function of mean 0 and std 1
        :return: Tracker2 object
        """
        def index_func(x):
            avg = x.Sa/x.Na
            idx = np.zeros(self.nb_arms)
            for k in range(self.nb_arms):
                s = int(x.Na[k]) + 2
                e = np.zeros(s)
                e[:-2] = np.array(x.rewards_arm[k])-avg[k]
                e[-2] = np.sqrt(s) * sigma
                e[-1] = -np.sqrt(s) * sigma
                w = weight_func(size=s)
                idx[k] = avg[k]+np.mean(w*e)
            return idx
        return self.Index_Policy(T, index_func, store_rewards_arm=True)

    def ReBootG(self, T, sigma):
        """
        More efficient version of ReBoot with the gaussian bootstrap
        :param T: Time Horizon
        :param sigma: standard deviation of perturbation
        :return: Tracker2 object
        """
        def index_func(x):
            avg = x.Sa/x.Na
            idx = np.zeros(self.nb_arms)
            for k in range(self.nb_arms):
                s = int(x.Na[k]) + 2
                e = np.zeros(s)
                e[:-2] = np.array(x.rewards_arm[k])-avg[k]
                e[-2] = np.sqrt(s) * sigma
                e[-1] = -np.sqrt(s) * sigma
                idx[k] = avg[k]+np.random.normal(loc=0, scale=1/(e.shape[0])*np.sqrt((e**2).sum()))
            return idx
        return self.Index_Policy(T, index_func, store_rewards_arm=True)

    def IMED(self, T):
        """
        Implementation of the IMED algorithm
        :param T: Time Horizon 
        :return: Tracker2 object
        """
        def index_func(x):
            mu_max = np.max(x.Sa/x.Na)
            idx = []
            for k in range(self.nb_arms):
                idx.append(x.Na[k]*self.kl(x.Sa[k]/x.Na[k], mu_max)+np.log(x.Na[k]))
            return -np.array(idx)
        return self.Index_Policy(T, index_func)

    def Bootstrapped_TS(self, T, prior, M):
        """
        Implementation of the Bootstrapped Thompson Sampling (Osband et al., 2017)
        :param T: Time Horizon 
        :param prior: prior for the fake history
        :param M: number of fake samples at each step
        :return: Tracker2 object
        """
        # éventuellement rajouter l'algo de bootstrap en param. Pour l'instant: with replacement
        def index_func(x):
            idx = []
            for k in range(self.nb_arms):
                artificial_hist = list(prior(size=M))
                n_tot = int(M + x.Na[k])
                bts_sample = np.random.choice(x.rewards_arm[k]+artificial_hist, replace=True, size=n_tot)
                idx.append(np.mean(bts_sample))
            return np.array(idx)
        return self.Index_Policy(T, index_func, store_rewards_arm=True)