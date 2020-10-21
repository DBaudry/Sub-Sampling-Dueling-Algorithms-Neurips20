from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle as pkl
from time import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from GaussianMAB import GaussianMAB
from BernoulliMAB import BetaBernoulliMAB
from ExponentialMAB import ExponentialMAB
from Trunc_GaussianMAB import TruncGaussianMAB
from tqdm import tqdm

mapping = {'B': BetaBernoulliMAB, 'G': GaussianMAB, 'Exp': ExponentialMAB, 'TG': TruncGaussianMAB}
mapping_name = {'B': 'Bernoulli', 'G': 'Gaussian', 'Exp': 'Exponential', 'TG': 'Truncated Gaussian'}

def MC_xp(args, plot=False, pickle_path=None, caption='xp'):
    """
    :param args: parameters of the experiment
    :param plot: Boolean, plot the average regret if True
    :param pickle_path: if not None, path to store the results
    :param caption: name of the file if pickle_path is not None
    :return: average regret, dict with all trajectories
    """
    bandit, p, T, n_xp, methods, param, store_step = args
    model = mapping[bandit](p)
    all_r = []
    all_traj = {}
    for x in methods:
        r, traj = model.MC_regret(x, n_xp, T, param[x], store_step)
        all_r.append(r)
        all_traj[x] = traj
    all_r.append(model.Cp*np.log(1+np.arange(T)))
    df_r = pd.DataFrame(all_r).T
    df_r.columns = methods + ['lower bound']
    df_r['lower bound'].iloc[0] = 0
    if plot:
        df_r.plot(figsize=(10, 8), logx=True)
    if pickle_path is not None:
        pkl.dump(df_r, open(os.path.join(pickle_path, caption+'.pkl'), 'wb'))
    return df_r, all_traj


def multiprocess_MC(args, plot=False, pickle_path=None, caption='xp'):
    """
    Same function as MC_xp, but including multiprocessing tools to allow parallelization
    """
    t0 = time()
    cpu = mp.cpu_count()
    print('Running on %i cores' % cpu)
    bandit, p, T, n_xp, methods, param, store_step = args
    new_args = (bandit, p, T, n_xp//cpu+1, methods, param, store_step)
    res = Parallel(n_jobs=cpu)(delayed(MC_xp)(new_args) for _ in range(cpu))
    df_r = res[0][0]
    for i in range(cpu-1):
        df_r += res[i+1][0]
    df_r = df_r/cpu
    traj = {}
    for x in methods:
        traj[x] = np.concatenate([res[i][1][x] for i in range(cpu)], axis=1)
    if plot:
        df_r.index = 1 + df_r.index
        df_r.plot(figsize=(10, 8), logx=True)
        plt.title('Average Regret for experiment ' + caption.split('_')[0] + ', ' + mapping_name[bandit] + ' arms (log scale)')
        plt.show()
    if pickle_path is not None:
        info = {'proba': p, 'N_xp': n_xp, 'T': T, 'methods': methods, 'param': param, 'step_traj': store_step}
        my_pkl_obj = {'df_regret': df_r, 'trajectories': traj, 'info': info}
        pkl.dump(my_pkl_obj, open(os.path.join(pickle_path, caption+'.pkl'), 'wb'))
    print('Execution time: %s seconds' % str(time()-t0))
    return df_r, traj


def Bayesian_MC_regret(args):
    """
    Implementation of Monte Carlo method to approximate the expectation of the regret
    :param method: list, methods used (UCB, Thomson Sampling, etc..)
    :param n_arms: number of arms for each experiment
    :param N: int, number of independent Monte Carlo simulation (one simul=one parameter)
    :param T: int, time horizon
    :param param_dic: dict, parameters for the different methods, can be the value of rho for UCB model or an int
    corresponding to the number of rounds of exploration for the ExploreCommit method
    """
    bandit, methods, n_arms, N, T, param, xp_sampler, step = args
    store_xp = np.zeros((len(methods), N, np.arange(T)[::step].shape[0]))
    mc_regret = pd.DataFrame(np.zeros((T, len(methods))), columns=methods)
    xp_list = []
    for n in tqdm(range(N)):
        p = xp_sampler(size=n_arms)
        xp_list.append(p)
        model = bandit(p)
        for i, m in enumerate(methods):
            alg = model.__getattribute__(m)
            tr = alg(T, **param[m])
            regret = tr.regret()
            mc_regret[m] += regret
            store_xp[i, n, :] = regret[::step]
    return {'regret': mc_regret/N, 'traj': store_xp, 'xp_list': xp_list}

def Bayesian_multiprocess_MC(args, pickle_path=None, plot=True, caption='xp'):
    """
    :param args:  parameters of the experiments
    :param pickle_path: If not None, path where the results are stored
    :param caption: Name of the file to store the results if pickle path is not none
    :return: dataframe of average regret, results for each trajectory/alg, xp settings
    """
    t0 = time()
    cpu = mp.cpu_count()
    print('Running on %i cores' % cpu)
    bandit, methods, n_arms, N, T, param, xp_sampler, step = args
    new_args = (bandit, methods, n_arms, N//cpu+1, T, param, xp_sampler, step)
    res = Parallel(n_jobs=cpu)(delayed(Bayesian_MC_regret)(new_args) for _ in range(cpu))

    df_r = res[0]['regret']
    xp_list = res[0]['xp_list']
    for i in range(cpu-1):
        df_r += res[i+1]['regret']
        xp_list += res[i+1]['xp_list']
    df_r = df_r/cpu
    traj = np.concatenate([res[i]['traj'] for i in range(cpu)], axis=1)
    if pickle_path is not None:
        info = {'type': 'Bayesian', 'N_xp': N, 'T': T, 'methods': methods, 'step_traj': step}
        my_pkl_obj = {'df_regret': df_r, 'trajectories': traj, 'info': info}
        pkl.dump(my_pkl_obj, open(os.path.join(pickle_path, caption+'.pkl'), 'wb'))
    if plot:
        df_r.index = 1 + df_r.index
        df_r.plot(figsize=(10, 8), logx=True)
        plt.title('Average Regret for bayesian experiment '+caption.split('_')[0] + ', ' + str(n_arms)+' arms (log scale)')
        plt.show()
    print('Execution time: %s seconds' % str(time()-t0))
    return df_r, traj, xp_list