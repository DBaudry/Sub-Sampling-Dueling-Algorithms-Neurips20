import numpy as np

class Tracker2:
    """
    This object is used in bandit models to store useful quantities to run the algorithm and report the experiment.
    """
    def __init__(self, means, T, store_rewards_arm=False):
        self.means = means
        self.nb_arms = means.shape[0]
        self.T = T
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms)
        self.reward = np.zeros(self.T)
        self.arm_sequence = np.empty(self.T, dtype=int)
        self.t = 0
        self.store_rewards_arm = store_rewards_arm
        if store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]

    def reset(self):
        """
        Initialization of quantities of interest used for all methods
        :param T: int, time horizon
        :return: - Sa: np.array, cumulative reward of arm a
                 - Na: np.array, number of times arm a has been pulled
                 - reward: np.array, rewards
                 - arm_sequence: np.array, arm chose at each step
        """
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms)
        self.reward = np.zeros(self.T)
        self.arm_sequence = np.zeros(self.T, dtype=int)
        self.rewards_arm = [[]]*self.nb_arms
        if self.store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]

    def update(self, t, arm, reward):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa:  np.array, cumulative reward array up to time t-1
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)

    def regret(self):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        return self.means.max() * np.arange(1, self.T + 1) - np.cumsum(np.array(self.means)[self.arm_sequence])