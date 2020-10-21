""" Packages import """
import numpy as np
from numba import jit
import bottleneck as bn
import scipy.stats as sc

@jit(nopython=True)
def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)


@jit(nopython=True)
def rd_choice(vec, size):
    """
    jit version of np.random.choice (slightly improve the computation time)
    """
    return np.random.choice(vec, size=size, replace=False)


@jit(nopython=True)
def hypergeom_sample(s1, n1, n2):
    """
    jit version of np.random.choice (slightly improve the computation time)
    """
    return np.random.hypergeometric(s1, n1 - s1, nsample=n2)


def rollavg_bottlneck(a, n):
    """
    :param a: array
    :param n: window of the rolling average
    :return: A fast function for computing moving averages
    """
    return bn.move_mean(a, window=n, min_count=n)


@jit(nopython=True)
def get_leader(Na, Sa, l_prev):
    """
    :param Na: Number of pulls of each arm (array)
    :param Sa: Sum of rewards of each arm (array)
    :param l_prev: previous leader
    :return: Leader for SSMC and SDA algorithms
    """
    m = np.amax(Na)
    n_argmax = np.nonzero(Na == m)[0]
    if n_argmax.shape[0] == 1:
        l = n_argmax[0]
        return l
    else:
        s_max = Sa[n_argmax].max()
        s_argmax = np.nonzero(Sa[n_argmax] == s_max)[0]
        if np.nonzero(n_argmax[s_argmax] == l_prev)[0].shape[0] > 0:
            return l_prev
    return n_argmax[np.random.choice(s_argmax)]


def get_SSMC_star_min(rewards_l, n_challenger, reshape_size):
    """
    little helper for SSMC*
    """
    return (np.array(rewards_l)[:n_challenger * reshape_size].reshape(
        (reshape_size, n_challenger))).mean(axis=1).min()


def convert_tg_mean(mu, scale, step=1e-7):
    """
    :param mu: mean of the underlying gaussian r.v
    :param scale: scale of the underlying gaussian r.v
    :param step: precision of the numerical integration
    :return: compute the mean of the Truncated Gaussian r.v knowing the parameters of its
    associated Gaussian r.v
    """
    X = np.arange(0, 1, step)
    return (X * sc.norm.pdf(X, loc=mu, scale=scale)).mean()+ 1 - sc.norm.cdf(1, loc=mu, scale=scale)