# Sub-Sampling Dueling Algorithms, Neurips 2020

This repositery contains the code associated with the paper 
"Sub-sampling for efficient Non-Parametric Bandit Exploration" presented at Neurips 2020. We provide a description of the code structure and a short guide to run some experiments.

## How to run experiments

The *__ main__.py* file contains different block with code that can be directly executed. This file relies on *xp_helpers.py*, that contains functions that 
allow to run two types of experiments: 
* Frequentist experiments: the user defines a bandit model and perform a number of runs
of each algorithm for this particular model
* Bayesian experiments: the user defines a prior distribution for the bandit model and draw a number of experiments from this distribution.
Then, each bandit algorithm runs once on these problems. 

The file is divided in three blocks. The __xp_type__ parameter allows to choose which block to run. Several examples are proposed in each blocks.

## Code Structure

### Bandit algorithms

Our implementation of the multi-arm bandit problem has its key structure in the *MAB.py* file. The initialization of the bandit relies on the *arms.py* file, which defines objects representing the arms and their properties (mean, how to sample the rewards, etc...). 

The __GenericMAB__ object is designed as a mother class for any bandit model. Several algorithms are already implemented in this class, when they don't have to be calibrated for specific distributions. The function __MC_regret__ allows to run a single bandit algorithm for a given number of runs and time horizon and returns the regret.

The objects __BernoulliMAB.py__, __GaussianMAB.py__, __ExponentialMAB.py__ and __TruncatedGaussianMAB.py__
are inherited from __GenericMAB__ and refine the class to adapt it to the Bernoulli, Gaussian, Exponential and Truncated Gaussian distributions. In particular,
they contain the algorithms that are specific to the family of distribution of the arms, or optimized versions of algorithms that are alerady in __GenericMAB__ (for instance in Bernoulli MAB). 

### Helpers 

The __Tracker2__ object defined *tracker.py* is a useful object used in all of our bandit algorithms to store the settings of the experiments during the runs.
In particular, it can be used to store the number of pulls, cumulated regret and reward history of each arm. 

*utils.py* contains several functions that are useful in the bandit algorithms. Some of these function use the *numba* package for faster computation. 

Finally, *xp_helpers.py* provide useful functions to perform large scale experiments in the frequentist and bayesian setting. Some of these functions use libraries that allow multiprocessing for parallel computation.
