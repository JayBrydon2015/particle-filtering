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
from scipy.linalg import expm # For computing the matrix exp: e^A

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
    ## Option 1 ##
    # return np.identity(n) + DELTA_T * lams_to_gen(lams)
    ## Option 2 ##
    return expm(DELTA_T * lams_to_gen(lams))


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
        R = np.exp( gammaln(n + mu / DELTA_T) - gammaln(mu / DELTA_T) )
        result = R * numerator_result / denominator_result
        # print(f"a: {a}")
        # print(f"b: {b}")
        # print(f"numerator: {numerator_result}")
        # print(f"denominator: {denominator_result}")
        # print(f"result: {result}")
        return result
    
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
            # print(f"idx: {idx}")
            # print(f"mu: {mu}")
            if np.isnan(mu):
                print(xp[:, idx])
                raise ValueError("mu is np.nan!")
            first_mom  = self.get_nth_moment(mu, 1, a, b)
            second_mom = self.get_nth_moment(mu, 2, a, b)
            var = second_mom - first_mom ** 2
            alpha, beta = get_gamma_params_from_mean_var(first_mom, var)
            lams_dists.append(dists.Gamma(alpha, beta))
        return dists.IndepProd(*lams_dists)


# %%

## Define time period ##
K = 300 # k = 0, 1, ..., K
k_series = np.arange(K + 1)
time_points = DELTA_T * k_series


## Number of particles for PFs ##
N = 1000


## Define SSM parameters ##
n = 2 # Number of states in CTMC
mu0 = np.array([2, 3])
var0 = np.array([0.2, 0.2])
assert mu0.shape == (n * (n-1),) and var0.shape == (n * (n-1),)
a0, b0 = get_gamma_params_from_mean_var(mu0, var0)
N_rws = 10 # Number of random walkers traversing the CTMC


## Create CTMC SSM(s) ##
ctmc_ssm = AugCTMC(a0=a0, b0=b0, n=n, N_rws=N_rws) # Boot PF
# ctmc_ssm_prop = AugCTMC_prop(a0=a0, b0=b0, n=n, N_rws=N_rws) # Guided PF


# %%

## Simulate true states and data manually ##

# Sigmoid growth #
true_states = [mu0.reshape(1, -1)]
data = [np.sort(np.array([j % n for j in range(N_rws)])).reshape(1, -1)]
for k in range(1, K+1):
    # lams
    lams_k = np.array([sigmoid(k, mu0[0], mu0[0]+1, K/2,
                               SIGMOID_FUNC_CONST / K),
                       sigmoid(k, mu0[1], mu0[1]+2, K/2,
                               SIGMOID_FUNC_CONST / K)])
    true_states.append(lams_k.reshape(1, -1))
    
    # y
    P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n)
                      for cur_lams in true_states[-2]], axis=0)
    y_t = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
                    for y_i in data[-1]])
    data.append(y_t)

# Constant rates #
# true_states = [mu0.reshape(1, -1) for _ in k_series]
# data = [ np.sort(np.array([j % n for j in range(N_rws)])).reshape(1, -1) ]
# for k in range(1, K+1):
#     # y
#     P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n)
#                       for cur_lams in true_states[k-1]], axis=0)
#     y_k = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
#                     for y_i in data[-1]])
#     # y_k = np.array([dists.Categorical(P_mat[:, y_i]).rvs()
#     #                 for y_i in data[-1]])
#     data.append(y_k)

# Linear growth #
# true_states = [mu0.reshape(1, -1)]
# data = [np.sort(np.array([j % n for j in range(N_rws)])).reshape(1, -1)]
# for k in range(1, K+1):
#     # lams
#     lams_k = np.array([mu0[0] + 2 * k / K,
#                        mu0[1], mu0[2], mu0[3], mu0[4],
#                        mu0[5] - k / K])
#     true_states.append(lams_k.reshape(1, -1))
    
#     # y
#     P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n)
#                       for cur_lams in true_states[-2]], axis=0)
#     y_t = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
#                     for y_i in data[-1]])
#     data.append(y_t)


## True lambdas dataframe ##
lams_gen_positions = [lams_idx_to_gen_pos(i, n)
                      for i in range(true_states[0].shape[1])]
true_lams = pd.DataFrame(np.stack([true_state.reshape(-1)
                                   for true_state in true_states]),
                         columns=[f"λ_{p}{q}" for p, q in lams_gen_positions],
                         index=k_series)
true_lams = true_lams.rename_axis('t')


# %%

## Plot the true lambdas ##

#plt.figure(figsize=(10, 4))

for col in true_lams.columns:
    plt.plot(k_series, true_lams[col], label=col)

plt.xlabel("k")
plt.ylabel("Value")
plt.title("True rates over time")
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
ds_boot = xr.Dataset({
    'X': xr.DataArray(
        np.stack([pf_boot.hist.X[k] for k in k_series]),
        dims=("k", "particle", "lam"),
        coords={
            "k": k_series,
            "lam": true_lams.columns.values,
        },
        name="Bootstrap PF Particles"
    ),
    'W': xr.DataArray(
        np.stack([pf_boot.hist.wgts[k].W for k in k_series]),
        dims=("k", "weight"),
        coords={
            "k": k_series
        },
        name="Bootstrap PF Weights"
    )
})


## Calculate quantiles and add into ds_boot ##

def weighted_quantile(values, weights, quantiles):
    """
    values: (particle,)
    weights: (particle,)
    quantiles: array-like in [0,1]
    """
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, values)

qs = np.array([0.05, 0.5, 0.95]) # 95% interval

ds_boot["X_quantiles"] = xr.apply_ufunc(
    weighted_quantile,
    ds_boot["X"],
    ds_boot["W"],
    input_core_dims=[["particle"], ["weight"]],
    output_core_dims=[["quantile"]],
    vectorize=True,
    kwargs={"quantiles": qs},
    dask="parallelized",
    output_dtypes=[float],
).assign_coords(quantile=qs)

# %%

## Guided PF ##

# fk_guided = augssm.AugmentedGuidedPF(ssm=ctmc_ssm_prop, data=data)
# pf_guided = particles.SMC(fk=fk_guided, N=N, resampling='stratified', 
#                           store_history=True, collect=[Moments()])
# pf_guided.run()

# # Store lambda particles in xarray.DataArray
# da_guided = xr.DataArray(
#     np.stack([pf_guided.hist.X[k] for k in k_series]),
#     dims=("k", "particle", "lam"),
#     coords={
#         "k": k_series,
#         "lam": true_lams.columns.values,
#     },
#     name="Guided PF Particles"
# )


# %%

## Band plots using quantiles: Boot ##

for lam_idx, lam in enumerate(true_lams.columns):
    median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
    lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
    uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
    plt.plot(k_series, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(median, color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(k_series, 
                     y1=lq, 
                     y2=uq, 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Value")
    plt.title(f"Boot PF band plot quantiles: {lam} | "
              + f"J={N_rws} N={N} DT={DELTA_T} C={C}")
    plt.show()


# %%

## Band plots naive: Boot ##

means_boot =  np.stack([m['mean'] for m in pf_boot.summaries.moments])
vars_boot = np.stack([m['var'] for m in pf_boot.summaries.moments])

for lam_idx, lam in enumerate(true_lams.columns):
    plt.plot(k_series, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(means_boot[..., lam_idx], color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(k_series, 
                     y1=(means_boot[..., lam_idx]
                         -2*np.sqrt(vars_boot[..., lam_idx])), 
                     y2=(means_boot[..., lam_idx]
                         +2*np.sqrt(vars_boot[..., lam_idx])), 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Value")
    plt.title(f"Boot PF band plot (old way): {lam} | "
              + f"J={N_rws} N={N} DT={DELTA_T} C={C}")
    plt.show()


# %%

## Band plots: Guided ##

# means_guided =  np.stack([m['mean'] for m in pf_guided.summaries.moments])
# vars_guided = np.stack([m['var'] for m in pf_guided.summaries.moments])

# for lam_idx, lam in enumerate(true_lams.columns):
#     plt.plot(k_series, true_lams[lam].values, label=f"True {lam}",
#              color='red', alpha=0.7)
#     plt.plot(means_guided[..., lam_idx], color="green",
#              label="PF mean", alpha=0.7)
#     plt.fill_between(k_series, 
#                      y1=(means_guided[..., lam_idx]
#                          -2*np.sqrt(vars_guided[..., lam_idx])), 
#                      y2=(means_guided[..., lam_idx]
#                          +2*np.sqrt(vars_guided[..., lam_idx])), 
#                      color="green", alpha=0.3)
#     plt.legend()
#     plt.xlabel("k")
#     plt.ylabel("Value")
#     plt.title(f"Guided PF band plot: {lam} | "
#               + f"J={N_rws} N={N} DT={DELTA_T} C={C}")
#     plt.show()


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
    fig.suptitle("RW states over time", fontsize=14)
    
    # Ensure axes is always iterable (important if n == 1)
    if N_rws == 1:
        axes = [axes]
    
    for i in range(N_rws):
        axes[i].plot(k_series, data_plot[:, i])
        axes[i].set_ylabel(f"RW #{i+1}")
        axes[i].grid(True)
    
    axes[-1].set_xlabel("k")
    plt.tight_layout()
    plt.show()
else:
    print(f"Too many random walkers to plot: {N_rws} RWs.")

# %%

## ESS: Boot ##

plt.plot(k_series, pf_boot.summaries.ESSs, color="red")
plt.xlabel("k")
plt.ylabel("ESS")
plt.title("ESS over time: Boot PF | "
          + f"J={N_rws} N={N} DT={DELTA_T} C={C}")
plt.show()

# %%

## ESS: Guided ##

# plt.plot(k_series, pf_guided.summaries.ESSs, color="red")
# plt.xlabel("k")
# plt.ylabel("ESS")
# plt.title("ESS over time: Guided PF | "
#           + f"J={N_rws} N={N} DT={DELTA_T} C={C}")
# plt.show()


# %%

## Select random t between 0 and T+1 ##

k = np.random.randint(K+1)


## KDE at k ##

# Boot + Guided #

# for lam in true_lams.columns:
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.kdeplot(x=da_boot.sel({'lam': lam, 'k': k}).values.reshape(-1),
#                weights=pf_boot.hist.wgts[k].W, ax=ax, fill=True,
#                color="skyblue", label="Boot")
#     # sns.kdeplot(x=da_guided.sel({'lam': lam, 'k': k}).values.reshape(-1),
#     #            weights=pf_guided.hist.wgts[k].W, ax=ax, fill=True,
#     #            color="lightcoral", label="Guided")
#     ax.axvline(x=true_lams.loc[k][lam], color='red', linestyle=':', linewidth=1.5, 
#                label='True state')
#     ax.set_xlabel("Value")
#     ax.set_ylabel("Density")
#     # ax.set_title(f"Boot + Guided Filtering Dist @ k = {k}: {lam}")
#     ax.set_title(f"Boot Filtering Dist @ k = {k}: {lam}")
#     ax.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.show()


## Pairwise scatter plots at k ##

# Boot #

# plot_df = (
#     da_boot.sel(k=k)
#     .to_pandas() # index: particle, columns: lambda
#     .reset_index(drop=True)
# )
# sns.pairplot(
#     plot_df,
#     plot_kws={"alpha": 0.5, "s": 15},
#     diag_kind="kde"
# )
# plt.suptitle(f"Pairwise scatter at k = {k}: Boot", y=1.02)
# plt.show()

# Guided #

# plot_df = (
#     da_guided.sel(k=k)
#     .to_pandas() # index: particle, columns: lambda
#     .reset_index(drop=True)
# )
# sns.pairplot(
#     plot_df,
#     plot_kws={"alpha": 0.5, "s": 15},
#     diag_kind="kde"
# )
# plt.suptitle(f"Pairwise scatter at k = {k}: Guided", y=1.02)
# plt.show()
