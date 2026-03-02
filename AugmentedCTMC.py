# %%

# -*- coding: utf-8 -*-

""" PF for inferring rates of CTMC """

## Imports ##

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr
from math import comb
from scipy.special import gammaln

# Particles package
import particles
from particles import augmented_state_space_models as augssm
from particles import distributions as dists
from particles.collectors import Moments


## CONSTANTS ##
DELTA_T = 0.02 # Time between observations. Keep small enough.
C = 1 # Scale the transition variance by C
SIGMOID_FUNC_CONST = 14

## Functions ##

def sigmoid(x, a, b, m, k):
    """ Result = a when t = 0. """
    result = a + (b-a) / (1 + np.exp(-k*(x-m)))
    result -= a + (b-a) / (1 + np.exp(k * m))
    result += a
    return result

def get_gamma_params_from_mean_var(mean, var):
    """ Compute the Gamma distribution parameters, alpha
        and beta, from the mean and variance. """
    return mean ** 2 / var, mean / var

# lams are assumed to be a list of rates ordered by
# 11, 12, 13, ..., 1n, 21, 23, ..., 2n, ..., (n-1)n
def gen_to_lams(gen):
    """ Convert generator A to lams.
        Essentially flattens and removes the diagonal elements.
        Can also be used to convert Gamma dist parameters if their
        of the same shape as the generator. """
    n = gen.shape[0]
    lams = np.array([])
    for m in range(n):
      lams = np.append(lams, gen[m,0:m])
      lams = np.append(lams, gen[m,m+1:])
    return lams

def lams_to_gen(lams):
    """ Convert lams to generator A. """
    l = len(lams)
    n = int((1 + np.sqrt(1 + 4*l))/2)
    gen = np.zeros((n,n))
    for m in range(n):
      lams_m = lams[m*(n-1):(m+1)*(n-1)]
      gen[m, 0:m] = lams_m[0:m]
      gen[m, m+1:] = lams_m[m:]
      gen[m,m] = - np.sum(lams_m)
    return gen

def lams_idx_to_gen_pos(idx, n):
    """ Returns the i, j position of the lambda given its index
        in the lams array and the number of CMTC states n. """
    i = idx // (n-1)
    j = idx %  (n-1)
    if i <= j:
        j += 1
    return i, j

def compute_transition_prob_matrix(lams, n):
    return np.identity(n) + DELTA_T * lams_to_gen(lams)

def get_obs(t):
    """ Returns true if an observation is aquired at this time. """
    return True


## Augmented SSM Classes ##


class AugCTMC(augssm.AugmentedStateSpaceModel):
    """ CTMC Augmented SSM

        ----- Parameters -----
        n: number of states
        N_rws: number of random walkers
        a0: Gamma dist alpha parameters for PX0 in (n, n) ndarray
        b0: Gamma dist beta parameters for PX0 in (n, n) ndarray

        ----- Notes -----
        - Track lams_list rather than generator A
        - y defined as in type 4
        - SSM starts with the RWs spread across the states as evenly as
          possible.
        - y: ndarray of shape (n*N_rws, )
        - lams: ndarray of shape (n*(n-1), )
    """

    def PX0(self):
        if len(self.a0.shape) > 1: # Currently a (n, n) ndarray
            self.a0 = gen_to_lams(self.a0)
            self.b0 = gen_to_lams(self.b0)
        lams_dists = [dists.Gamma(a, b) for a, b in zip(self.a0, self.b0)]
        return dists.IndepProd(*lams_dists)
    
    def PX(self, t, xp):
        ## A ##
        
        # Option 1: Var = lambda * DELTA_T * C
        alpha = xp / DELTA_T
        beta_i = np.repeat(1/DELTA_T, alpha.shape[0])
        alpha  /= C
        beta_i /= C
        lams_dists = [dists.Gamma(alpha[:,l], beta_i)
                      for l in range(alpha.shape[1])]
        
        # Option 2: Var = DELTA_T * C
        # alpha = xp["lams"] ** 2 / DELTA_T
        # beta = xp["lams"] / DELTA_T
        # alpha /= C
        # beta  /= C
        # lams_dists = [dists.Gamma(alpha[:,l], beta[:, l])
        #               for l in range(alpha.shape[1])]
        
        return dists.IndepProd(*lams_dists)
    
    def get_cat_dist(self, P_mat, y_i):
        return dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i])
        # return dists.Categorical(P_mat[:, y_i])

    def PY(self, t, xp, x, datap=None):
        ## y ##
        
        if t == 0:
            y0 = np.sort(np.array([i % self.n for i in range(self.N_rws)]))
            y0 = np.stack([y0 for _ in range(x.shape[0])], axis=0)
            y0_dists = [dists.DiscreteDirac(y0[..., r])
                        for r in range(self.N_rws)]
            return dists.IndepProd(*y0_dists)
        
        else: # t >= 1
            P_mat = np.stack([compute_transition_prob_matrix(cur_lams, self.n)
                              for cur_lams in xp], axis=0)
            y_dists = [self.get_cat_dist(P_mat, datap[:, i])
                                for i in range(datap.shape[1])]
            return dists.IndepProd(*y_dists)


class AugCTMC_prop(AugCTMC):
    """ CTMC Augmented SSM with proposal. """
    
    def proposal0(self, data):
        return self.PX0()
    
    def compute_transition_count(self, datap, data):
        return np.bincount(self.n * datap.reshape(-1) + data.reshape(-1),
                minlength=self.n * self.n).reshape(self.n, self.n)
    
    def compute_numerator_or_denominator(self, mu, n, a, b, m):
        return ((-1) ** m * comb(a, m) *
                (1 / DELTA_T + DELTA_T * (b + m)) ** (-mu / DELTA_T - n))
    
    def get_nth_moment(self, mu, n, a, b):
        numerator_result = sum(
            self.compute_numerator_or_denominator(mu, n, a, b, m)
            for m in range(a+1)
        )
        denominator_result = sum(
            self.compute_numerator_or_denominator(mu, 0, a, b, m)
            for m in range(a+1)
        )
        R = np.exp( gammaln((mu + n) / DELTA_T) - gammaln(mu / DELTA_T) )
        return R * numerator_result / denominator_result
    
    def proposal(self, t, xp, data):
        lams_means = np.mean(xp, axis=0)
        trans_count_mat = self.compute_transition_count(data[t-1], data[t])
        lams_dists = []
        for idx, mu in enumerate(lams_means):
            p, q = lams_idx_to_gen_pos(idx, self.n)
            # Compute a & b
            a = trans_count_mat[p, q]
            b = trans_count_mat[p, p]
            # Compute first and second moments
            first_mom  = self.get_nth_moment(mu, 1, a, b)
            second_mom = self.get_nth_moment(mu, 2, a, b)
            var = second_mom - first_mom ** 2
            lams_dists.append(dists.Gamma(first_mom ** 2 / var,
                                          first_mom / var))
        return dists.IndepProd(*lams_dists)


# %%

## Define time period ##
T = 300 # t = 0, 1, ..., T
t_obs = [t for t in range(T+1) if get_obs(t)]
t_system = range(T+1)
true_time = np.array([DELTA_T*i for i in range(T+1)])


## Number of particles for PFs ##
N = 1000


## Define SSM parameters ##
n = 2 # Number of states in CTMC
mu0 = np.array([2, 3])
var0 = np.array([0.2, 0.2])
a0, b0 = get_gamma_params_from_mean_var(mu0, var0)
N_rws = 2 # Number of random walkers traversing the CTMC


## Create CTMC SSM ##
ctmc_ssm = AugCTMC(a0=a0, b0=b0, n=n, N_rws=N_rws)
ctmc_ssm_prop = AugCTMC_prop(a0=a0, b0=b0, n=n, N_rws=N_rws)


# %%

## Simulate true states and data manually ##

# Sigmoid growth #
true_states = [mu0.reshape(1, -1)]
data = [np.sort(np.array([i % n for i in range(N_rws)])).reshape(1, -1)]
for t in range(1, T+1):
    # lams
    lams_t = np.array([sigmoid(t, mu0[0], mu0[0]+1, T/2,
                               SIGMOID_FUNC_CONST / T),
                       sigmoid(t, mu0[1], mu0[1]+2, T/2,
                               SIGMOID_FUNC_CONST / T)])
    true_states.append(lams_t.reshape(1, -1))
    
    # y
    P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n)
                      for cur_lams in true_states[-2]], axis=0)
    y_t = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
                    for y_i in data[-1]])
    data.append(y_t)

# Constant rates #
# true_states = [mu0.reshape(1, -1) for _ in range(T+1)]
# data = [np.sort(np.array([i % n for i in range(N_rws)])).reshape(1, -1)]
# for t in range(1, T+1):
#     # y
#     P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n)
#                       for cur_lams in true_states[-2]], axis=0)
#     y_t = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
#                     for y_i in data[-1]])
#     data.append(y_t)

# Linear growth #
# ...


## True lambdas dataframe ##
lams_gen_positions = [lams_idx_to_gen_pos(i, n)
                      for i in range(true_states[0].shape[1])]
true_lams = pd.DataFrame(np.stack([true_state.reshape(-1)
                                   for true_state in true_states]),
                         columns=[f"λ_{p}{q}" for p, q in lams_gen_positions],
                         index=t_system)
true_lams = true_lams.rename_axis('t')


# %%

## Plot the true lambdas ##

#plt.figure(figsize=(10, 4))

for col in true_lams.columns:
    plt.plot(t_system, true_lams[col], label=col)

plt.xlabel("t")
plt.ylabel("Value")
plt.title("True lambdas")
plt.legend()
plt.tight_layout()
plt.show()

# %%

## Bootstrap PF ##

fk_boot = augssm.AugmentedBootstrap(ssm=ctmc_ssm, data=data)
pf_boot = particles.SMC(fk=fk_boot, N=N, resampling='stratified', 
                        store_history=True, collect=[Moments()])
pf_boot.run()

# Store lambda particles in xarray.DataArray
da_boot = xr.DataArray(
    np.stack([pf_boot.hist.X[t] for t in t_system]),
    dims=("time", "particle", "lambda"),
    coords={
        "time": t_system,
        "lambda": true_lams.columns.values,
    },
    name="Bootstrap Particles"
)


# %%

## Guided PF ##

fk_guided = augssm.AugmentedGuidedPF(ssm=ctmc_ssm_prop, data=data)
pf_guided = particles.SMC(fk=fk_guided, N=N, resampling='stratified', 
                          store_history=True, collect=[Moments()])
pf_guided.run()

# Store lambda particles in xarray.DataArray
da_guided = xr.DataArray(
    np.stack([pf_guided.hist.X[t] for t in t_system]),
    dims=("time", "particle", "lambda"),
    coords={
        "time": t_system,
        "lambda": true_lams.columns.values,
    },
    name="Bootstrap Particles"
)


# %%

## Band plots: Boot ##

means_boot =  np.stack([m['mean'] for m in pf_boot.summaries.moments])
vars_boot = np.stack([m['var'] for m in pf_boot.summaries.moments])

for lam_idx, lam in enumerate(true_lams.columns):
    plt.plot(t_system, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(means_boot[..., lam_idx], color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(t_system, 
                     y1=(means_boot[..., lam_idx]
                         -2*np.sqrt(vars_boot[..., lam_idx])), 
                     y2=(means_boot[..., lam_idx]
                         +2*np.sqrt(vars_boot[..., lam_idx])), 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title(f"Augmented Boot PF band plot: {lam} | "
              + f"N_RW={N_rws} N={N} DT={DELTA_T} C={C}")
    plt.show()


# %%

## Band plots: Guided ##

means_guided =  np.stack([m['mean'] for m in pf_guided.summaries.moments])
vars_guided = np.stack([m['var'] for m in pf_guided.summaries.moments])

for lam_idx, lam in enumerate(true_lams.columns):
    plt.plot(t_system, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(means_guided[..., lam_idx], color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(t_system, 
                     y1=(means_guided[..., lam_idx]
                         -2*np.sqrt(vars_guided[..., lam_idx])), 
                     y2=(means_guided[..., lam_idx]
                         +2*np.sqrt(vars_guided[..., lam_idx])), 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title(f"Augmented Guided PF band plot: {lam} | "
              + f"N_RW={N_rws} N={N} DT={DELTA_T} C={C}")
    plt.show()


# %%

## Plot data over time ##

if N_rws <= 10:
    data_plot = np.vstack(data)
    
    fig, axes = plt.subplots(
        nrows=N_rws,
        ncols=1,
        sharex=True,
        figsize=(8, 2.5 * n)
    )
    fig.suptitle("RW States Over Time", fontsize=14)
    
    # Ensure axes is always iterable (important if n == 1)
    if N_rws == 1:
        axes = [axes]
    
    for i in range(N_rws):
        axes[i].plot(t_system, data_plot[:, i])
        axes[i].set_ylabel(f"RW #{i+1}")
        axes[i].grid(True)
    
    axes[-1].set_xlabel("t")
    plt.tight_layout()
    plt.show()
else:
    print(f"Too many random walkers to plot: {N_rws} RWs.")

# %%

## ESS: Boot ##

plt.plot(t_system, pf_boot.summaries.ESSs, color="red")
plt.xlabel("t")
plt.ylabel("ESS")
plt.title("ESS Over Time: Boot PF | "
          + f"N_RW={N_rws} N={N} DT={DELTA_T} C={C}")
plt.show()


## ESS: Guided ##

plt.plot(t_system, pf_guided.summaries.ESSs, color="red")
plt.xlabel("t")
plt.ylabel("ESS")
plt.title("ESS Over Time: Guided PF | "
          + f"N_RW={N_rws} N={N} DT={DELTA_T} C={C}")
plt.show()


# %%

## Select random t between 0 and T+1 ##

t = np.random.randint(T+1)


## KDE at t ##

# Boot + Guided #

for lam in true_lams.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(x=da_boot.sel({'lambda': lam})[t].values.reshape(-1),
               weights=pf_boot.hist.wgts[t].W, ax=ax, fill=True,
               color="skyblue", label="Boot")
    sns.kdeplot(x=da_guided.sel({'lambda': lam})[t].values.reshape(-1),
               weights=pf_guided.hist.wgts[t].W, ax=ax, fill=True,
               color="lightcoral", label="Guided")
    ax.axvline(x=true_lams.loc[t][lam], color='red', linestyle=':', linewidth=1.5, 
               label='True state')
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Boot + Guided Filtering Dist @ t = {t}: {lam}")
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


## Pairwise scatter plots at t ##

# Boot #

plot_df = (
    da_boot.sel(time=t)
    .to_pandas() # index: particle, columns: lambda
    .reset_index(drop=True)
)
sns.pairplot(
    plot_df,
    plot_kws={"alpha": 0.5, "s": 15},
    diag_kind="kde"
)
plt.suptitle(f"Pairwise scatter at time t = {t}: Boot", y=1.02)
plt.show()

# Guided #

plot_df = (
    da_guided.sel(time=t)
    .to_pandas() # index: particle, columns: lambda
    .reset_index(drop=True)
)
sns.pairplot(
    plot_df,
    plot_kws={"alpha": 0.5, "s": 15},
    diag_kind="kde"
)
plt.suptitle(f"Pairwise scatter at time t = {t}: Guided", y=1.02)
plt.show()
