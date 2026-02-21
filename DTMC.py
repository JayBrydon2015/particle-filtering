# %%

# -*- coding: utf-8 -*-

""" Running a PF on a DTMC """

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats

# Particles package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments

# Constants


def get_obs(t):
    return t > 0

class DTMC(ssm.StateSpaceModel):
    """ DTMC SSM """
    
    def OneHot(self, Y): # Will debug later
        result = np.zeros(self.NumStates)
        result[Y] = 1
        return result
    
    def PX0(self):
        # Self.P = ... ?
        return dists.StructDist({"P00": dists.Uniform(self.a00, self.b00),
                                 "P11": dists.Uniform(self.a11, self.b11),
                                 "Y": dists.Dirac(0)})
    
    def PX(self, t, xp):
        P00 = xp["P00"]
        P11 = xp["P11"]
        Y = xp["Y"]
        P_new = np.array([[P00, 1-P00], [P11, 1-P11]])
        y_dist = dists.Categorical(P_new.reshape(2,2) @ self.OneHot(Y.astype(int)[0]))
        return dists.StructDist({"P00": dists.Dirac(P00),
                                 "P11": dists.Dirac(P11),
                                 "Y": y_dist})
    
    def PY(self, t, xp, x):
        return dists.Dirac(x["Y"])


a00 = 0.3
a11 = 0.4
b00 = 0.7
b11 = 0.8

T = 10 # t = 0, 1, ..., T
t_obs = [t for t in range(T+1) if get_obs(t)]
t_system = range(T+1)

## Number of particles for PFs ##
N = 300

dtmc_ssm = DTMC(a00 = a00, b00 = b00, a11 = a11, b11 = b11, NumStates = 2)

true_states, data = dtmc_ssm.simulate(T+1)
data_clean = [val for val in data if not np.isnan(val)] # For plotting

# %%
