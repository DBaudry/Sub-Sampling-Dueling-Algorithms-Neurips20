from BernoulliMAB import BetaBernoulliMAB
from GaussianMAB import GaussianMAB
from ExponentialMAB import ExponentialMAB
from Trunc_GaussianMAB import TruncGaussianMAB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xp_helpers import multiprocess_MC, Bayesian_MC_regret, Bayesian_multiprocess_MC
from time import time
import pickle as pkl
import os

results_path = "xp_results"

xp_type = "test"
#xp_type = "frequentist"
#xp_type = "bayesian"

# Enter the parameters of each algorithms
param = {'SSMC': {}, 'WR_SDA': {}, 'RB_SDA': {}, 'BESA': {},
         'TS': {}, 'kl_ucb': {'f': np.log}, 'PHE': {'a': 1.1}, 'vanilla_bootstrap': {}, 'LB_SDA': {},
         'LDS_SDA': {}, 'non_parametric_TS': {}, 'IMED': {}, 'Bootstrapped_TS': {'M': 10, 'prior': np.random.random},
         'ReBoot': {'sigma': 1}, 'ReBootG': {'sigma': 2.}}

if __name__ == '__main__' and xp_type == "frequentist":
    # Settings of the experiments of the paper
    xp_bernoulli = {'xp1': [0.9, 0.8], 'xp2': [0.5, 0.6], 'xp3': [0.01]*3+[0.03]*3+[0.1]+[0.05]*3,
                    'xp4': [0.85]*7+[0.9]}
    xp_gaussian = {'xp1': [[0., 1.], [0.5, 1.]], 'xp2': [[0., 1.], [0., 1.], [0., 1.], [0.5, 1.]],
                   'xp3': [[0., 1.], [0.5, 1.], [1.0, 1.], [1.5, 1.]]}
    xp_expo = {'xp1': [1, 1.5], 'xp2': [0.1, 0.2], 'xp3': [10, 11], 'xp4': [1, 2, 3, 4],
               'xp5': [0.1, 0.2, 0.3, 0.4], 'xp6': [4, 4, 4, 5]}
    xp_TG = {'xp1': [[0.5, 0.1], [0.6, 0.1]], 'xp2': [[0., 0.3], [0.2, 0.3]],
            'xp3': [[1.5, 1.], [2., 1.]], 'xp4': [[0.4, 1.], [0.5, 1.], [0.6, 1.], [0.7, 1.]]}
    xp_settings = {'TG': xp_TG, 'G': xp_gaussian, 'Exp': xp_expo, 'B': xp_bernoulli}

    # General Parameters
    algs = ['TS', 'RB_SDA', 'WR_SDA', 'LB_SDA', 'BESA', 'SSMC']  # Select some Algorithms (check param file for availability)
    T, N = 1000, 100  # Time Horizon and Number of runs
    step = 25  # If results are saved trajectories are stored for all rounds such that t%step=0

    # Run
    xp_family = 'B'
    # xp = [(xp_family, x) for x in xp_settings[xp_family]] # To run all xp defined for a family
    xp = [('B', 'xp1'), ('G', 'xp2'), ('Exp', 'xp3'), ('TG', 'xp4')]  # To run any subset of xp
    for x in xp:
        caption = x[1] + '_' + x[0] + str(int(np.random.uniform() * 1e6))  # Name of the results file
        print(caption)  # caption=None allow to avoid saving the results
        res, traj = multiprocess_MC((x[0], xp_settings[x[0]][x[1]], T, N,
                                     algs, param, step), plot=True, pickle_path=results_path, caption=caption)


if __name__ == "__main__" and xp_type == "bayesian":
    # Possible Samplers for the experiments
    def sp_xp_B(size):  # Generate means uniformly in [0, 1]
        return np.random.uniform(0., 1., size=size)

    def sp_xp_G(size): # Generate means with a gaussian distribution and add a variance param of 1
        a = np.random.normal(0., 1., size=size)
        return [[x, 1] for x in a]

    # General Parameters
    n_arms = 3  # number of arms
    bandit = GaussianMAB  # bandit model (object inherited from class MAB)
    xp_sampler = sp_xp_G  # function to generate the problems
    N, T = 100, 2000  # Number of problems generate and Time Horizon of each run
    step = 100
    algs = ['TS', 'BESA', 'RB_SDA', 'WR_SDA', 'SSMC']
    args = (bandit, algs, n_arms, N, T, param, xp_sampler, step)  # do not modify
    caption = 'Gaussian_' + str(np.random.randint(1e5))
    b = Bayesian_multiprocess_MC(args, pickle_path=results_path, caption=caption)

if __name__ == "__main__" and xp_type == "test":
    # # Test anything here
    # model = TruncGaussianMAB([[0.5, 1], [0.6, 1]])
    # print(model.MC_regret(method='RB_SDA', N=500, T=20000, param_dic=param['WR_SDA']))
    #
    # # Example to load the pickles and read the results
    # name = 'xp1_B161845.pkl'
    # res = pkl.load(open(os.path.join(results_path, name), 'rb'))
    # print(res['info'])  # parameters dic
    # # Working with every runs of some algorithm
    # print(res['trajectories']['BESA'][-1].mean(), res['trajectories']['BESA'][-1].std())
    # # The average regret dataframe
    # res['df_regret'].plot()
    # plt.show()
    model = GaussianMAB([[0., 1], [0, 1]])
    res = model.RB_SDA(T=2000)
    n = np.array([np.cumsum(res.arm_sequence == i) for i in range(2)]).T
    l = np.argmax(n, axis=1)
    count_change = np.sum([l[i] != l[i-1] for i in range(1, len(l))])
    count_l_draw = np.sum([np.sum((l==i)*0) for i in range(2)])
    rw = [np.cumsum(x)/(np.arange(len(x))+1) for x in res.rewards_arm]
    print(res)
