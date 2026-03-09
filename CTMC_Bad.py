# %%

# -*- coding: utf-8 -*-

""" PF for inferring rates of CTMC """

## Imports; functions; SSM classes ##

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr

# Particles package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments # Doesn't seem to work for StructDist?


## CONSTANTS ##
DELTA_T = 0.05 # Time between observations. Keep small enough.
C = 0.3 # Scale the transition variance by C


## Functions ##

def sigmoid(x, a, b, m, k):
    return a + (b-a) / (1 + np.exp(-k*(x-m)))

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
    j = idx % (n-1)
    if i <= j:
        j += 1
    return i, j

def compute_transition_prob_matrix(lams, n):
    return np.identity(n) + DELTA_T * lams_to_gen(lams)

def get_obs(t):
    """ Returns true if an observation is aquired at this time. """
    return True


## SSM Classes ##

class CTMC(ssm.StateSpaceModel):
    """ CTMC SSM

        ----- Parameters -----
        n: number of states
        N_rws: number of random walkers
        a0: Gamma dist alpha parameters for PX0 in (n, n) ndarray
        b0: Gamma dist beta parameters for PX0 in (n, n) ndarray

        ----- Notes -----
        - Track lams_list rather than generator A
        - y defined as in type 4
        - SSM starts with the RWs either all in state 0, or spread across
          the states as evenly as possible.
        - y: ndarray of shape (n*N_rws, )
        - lams: ndarray of shape (n*(n-1), )
    """

    def PX0(self):
        a0_fl = gen_to_lams(self.a0)
        b0_fl = gen_to_lams(self.b0)
        lams_dists = np.array([dists.Gamma(a, b) for a, b in zip(a0_fl, b0_fl)])
        # y0 = np.repeat(0, self.N_rws)
        y0 = np.sort(np.array([i % self.n for i in range(self.N_rws)]))
        y0_dists = np.array([dists.DiscreteDirac(l) for l in y0])
        d = {
            "lams": dists.IndepProd(*lams_dists),
            "y": dists.IndepProd(*y0_dists)
        }
        return dists.StructDist(d)
    
    def get_cat_dist(self, P_mat, y_i):
        return dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i])
    
    def PX(self, t, xp):
        ## A ##
        # Option 1: Var = lambda*DELTA_T*C
        alpha = xp["lams"] / DELTA_T
        beta_i = np.repeat(1/DELTA_T, alpha.shape[0])
        alpha  /= C
        beta_i /= C
        lams_dists = np.array([dists.Gamma(alpha[:,l], beta_i)
                               for l in range(alpha.shape[1])])
        # Option 2: Var = DELTA_T
        # alpha = xp["lams"] ** 2 / DELTA_T
        # beta = xp["lams"] / DELTA_T
        # lams_dists = np.array([dists.Gamma(alpha[:,l], beta[:, l])
        #                        for l in range(alpha.shape[1])])
        
        ## y ##
        P_mat = np.stack([compute_transition_prob_matrix(cur_lams, self.n)
                          for cur_lams in xp["lams"]], axis=0)
        y_dists = [self.get_cat_dist(P_mat, xp["y"][:, i])
                            for i in range(xp["y"].shape[1])]
        d = {
            "lams": dists.IndepProd(*lams_dists),
            "y": dists.IndepProd(*y_dists)
        }
        return dists.StructDist(d)

    def PY(self, t, xp, x):
        return dists.IndepProd(*np.array([dists.DiscreteDirac(x["y"][:, i])
                                          for i in range(x["y"].shape[1])]))

class CTMC_prop_1(CTMC):
    """ CTMC with proposal #1. """
    
    def proposal0(self, data):
        # ## A ##
        # a0_fl = gen_to_lams(self.a0)
        # b0_fl = gen_to_lams(self.b0)
        # lams_dists = np.array([dists.Gamma(a, b) for a, b in zip(a0_fl, b0_fl)])
        # ## y ##
        # y_dists = [dists.DiscreteDirac(data[0][0][i])
        #            for i in range(self.N_rws)]
        # d = {
        #     "lams": dists.IndepProd(*lams_dists),
        #     "y": dists.IndepProd(*y_dists)
        # }
        # return dists.StructDist(d)
        return self.PX0()
    
    def proposal(self, t, xp, data):
        ## A ##
        lams_mean_est = np.mean(xp["lams"], axis=0)
        lams_var_est = np.var(xp["lams"], axis=0)
        # alpha and beta est (MoM) from sample
        alpha_ests = lams_mean_est ** 2 / lams_var_est
        beta_ests = lams_mean_est / lams_var_est
        lams_dists = [dists.Gamma(alpha_ests[i], beta_ests[i])
                      for i in range(xp["lams"].shape[1])]
        ## y ##
        y_dists = [dists.DiscreteDirac(data[t][0][i])
                   for i in range(self.N_rws)]
        d = {
            "lams": dists.IndepProd(*lams_dists),
            "y": dists.IndepProd(*y_dists)
        }
        if t % 50 == 0:
            print(t)
        return dists.StructDist(d)

class CTMC_prop_2(CTMC):
    """ CTMC with proposal #2. """
    
    def proposal0(self, data):
        return self.PX0()
    
    def proposal(self, t, xp, data):
        ## A ##
        alpha = xp["lams"] / DELTA_T
        beta_i = np.repeat(1/DELTA_T, alpha.shape[0])
        alpha  /= C
        beta_i /= C
        lams_dists = np.array([dists.Gamma(alpha[:,l], beta_i)
                               for l in range(alpha.shape[1])])
        # y ##
        y_dists = [dists.DiscreteDirac(data[t][0][i])
                   for i in range(self.N_rws)]
        d = {
            "lams": dists.IndepProd(*lams_dists),
            "y": dists.IndepProd(*y_dists)
        }
        if t % 50 == 0:
            print(t)
        return dists.StructDist(d)

# %%

## Define time period ##
T = 500 # t = 0, 1, ..., T
t_obs = [t for t in range(T+1) if get_obs(t)]
t_system = range(T+1)
true_time = np.array([DELTA_T*i for i in range(T+1)])


## Define SSM parameters ##

n = 2 # Number of states in CTMC
mu0 = np.array([[np.nan, sigmoid(0, 2, 3, T/2, 0.0015/DELTA_T)],
                [sigmoid(0, 3, 5, T/2, 0.0015/DELTA_T), np.nan]])
mu0 = np.array([[np.nan, 2],
                [2, np.nan]])
var0 = np.array([[np.nan, 0.1],
                 [0.15, np.nan]])
a0, b0 = get_gamma_params_from_mean_var(mu0, var0)
N_rws = 500 # Number of random walkers traversing the CTMC


## Number of particles for PFs ##
N = 1000


## Create the CTMC SSM and simulate from the CTMC class ##
ctmc_ssm_prop1 = CTMC_prop_1(a0=a0, b0=b0, n=n, N_rws=N_rws)
ctmc_ssm_prop2 = CTMC_prop_2(a0=a0, b0=b0, n=n, N_rws=N_rws)
# true_states, data = ctmc_ssm_prop1.simulate(T+1)

# %%

## Simulate true states and data manually ##
true_states = []
data = []
true_state_dtype = [
    ('lams', np.float64, (n*(n-1),)),
    ('y',    np.int64,   (N_rws,))
]
lams_0 = gen_to_lams(mu0) # Constant lams
# t = 0
cur_true_state = np.array(
    [(
        lams_0,
        np.sort(np.array([i % n for i in range(N_rws)]))
    )],
    dtype=true_state_dtype
)
true_states.append(cur_true_state)
data.append(true_states[-1]['y'])
# t >= 1
for t in range(1, T+1):
    # lams
    # lams_t = np.array([sigmoid(t, 2, 3, T/2, 0.0015/DELTA_T),
    #                    sigmoid(t, 3, 5, T/2, 0.0015/DELTA_T)])
    lams_t = np.array([lams_0[0], lams_0[0]+(t/200)])
    
    # y
    P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n)
                      for cur_lams in true_states[-1]['lams']], axis=0)
    y_t = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
                    for y_i in true_states[-1]['y']])
    
    cur_true_state = np.array(
        [(
            lams_t,
            y_t
        )],
        dtype=true_state_dtype
    )
    true_states.append(cur_true_state)
    data.append(true_states[-1]['y'])

## True lambdas dataframe ##
lams_gen_positions = [lams_idx_to_gen_pos(i, n)
                      for i in range(true_states[0]['lams'].shape[1])]
true_lams = pd.DataFrame(np.stack([true_state['lams'].reshape(-1)
                                   for true_state in true_states]),
                         columns=[f"λ_{p}{q}" for p, q in lams_gen_positions],
                         index=t_system)
true_lams = true_lams.rename_axis('t')

# %%

## Plot the true lambdas ##

#plt.figure(figsize=(10, 4))

for col in true_lams.columns:
    plt.plot(true_time, true_lams[col], label=col)

plt.xlabel("t")
plt.ylabel("Value")
plt.title("True lambdas")
plt.legend()
plt.tight_layout()
plt.show()

# %%

## Bootstrap PF ##

# fk_boot = ssm.Bootstrap(ssm=ctmc_ssm_prop1, data=data)
# pf_boot = particles.SMC(fk=fk_boot, N=N, resampling='stratified', 
#                         store_history=True) #, collect=[Moments()])
# pf_boot.run()

# # Store lambda particles in xarray.DataArray
# da_boot = xr.DataArray(
#     np.stack([pf_boot.hist.X[t]['lams'] for t in t_system]),
#     dims=("time", "particle", "lambda"),
#     coords={
#         "time": t_system,
#         "lambda": true_lams.columns.values,
#     },
#     name="Bootstrap Particles"
# )

# %%

## Guided PF: proposal #1 ##

fk_guided_p1 = ssm.GuidedPF(ssm=ctmc_ssm_prop1, data=data)
pf_guided_p1 = particles.SMC(fk=fk_guided_p1, N=N, resampling='stratified', 
                          store_history=True) #, collect=[Moments()])
pf_guided_p1.run()

# Store lambda particles in xarray.DataArray
da_guided_p1 = xr.DataArray(
    np.stack([pf_guided_p1.hist.X[t]['lams'] for t in t_system]),
    dims=("time", "particle", "lambda"),
    coords={
        "time": t_system,
        "lambda": true_lams.columns.values,
    },
    name="Guided Particles"
)

# %%

## Guided PF: proposal #2 ##

fk_guided_p2 = ssm.GuidedPF(ssm=ctmc_ssm_prop2, data=data)
pf_guided_p2 = particles.SMC(fk=fk_guided_p2, N=N, resampling='stratified', 
                          store_history=True) #, collect=[Moments()])
pf_guided_p2.run()

# Store lambda particles in xarray.DataArray
da_guided_p2 = xr.DataArray(
    np.stack([pf_guided_p2.hist.X[t]['lams'] for t in t_system]),
    dims=("time", "particle", "lambda"),
    coords={
        "time": t_system,
        "lambda": true_lams.columns.values,
    },
    name="Guided Particles"
)

# %%

## Collect mean and var of lams: guided prop 1 ##

L = da_guided_p1.sizes["lambda"]

mean = np.zeros((T+1, L))
var  = np.zeros((T+1, L))

for t in range(T+1):
    # particles at time t: shape (particle, lambda)
    x = da_guided_p1.isel(time=t).values

    # weights: shape (particle,)
    w = pf_guided_p1.hist.wgts[t].W

    # weighted mean: (lambda,)
    mu = np.sum(w[:, None] * x, axis=0)

    # weighted variance: (lambda,)
    v = np.sum(w[:, None] * (x - mu)**2, axis=0)

    mean[t] = mu
    var[t] = v

ds_guided_p1_moments = xr.Dataset(
    {
        "mean": (("time", "lambda"), mean),
        "var":  (("time", "lambda"), var),
    },
    coords={
        "time": da_guided_p1.time,
        "lambda": da_guided_p1["lambda"],
    }
)

## Collect ESS: guided ##

ess_guided_p1 = np.zeros(T+1)

for t in range(T+1):
    w = pf_guided_p1.hist.wgts[t].W
    ess_guided_p1[t] = 1.0 / np.sum(w**2)


## Collect mean and var of lams: guided prop 2 ##

L = da_guided_p2.sizes["lambda"]

mean = np.zeros((T+1, L))
var  = np.zeros((T+1, L))

for t in range(T+1):
    # particles at time t: shape (particle, lambda)
    x = da_guided_p2.isel(time=t).values

    # weights: shape (particle,)
    w = pf_guided_p2.hist.wgts[t].W

    # weighted mean: (lambda,)
    mu = np.sum(w[:, None] * x, axis=0)

    # weighted variance: (lambda,)
    v = np.sum(w[:, None] * (x - mu)**2, axis=0)

    mean[t] = mu
    var[t] = v

ds_guided_p2_moments = xr.Dataset(
    {
        "mean": (("time", "lambda"), mean),
        "var":  (("time", "lambda"), var),
    },
    coords={
        "time": da_guided_p2.time,
        "lambda": da_guided_p2["lambda"],
    }
)

## Collect ESS: guided ##

ess_guided_p2 = np.zeros(T+1)

for t in range(T+1):
    w = pf_guided_p2.hist.wgts[t].W
    ess_guided_p2[t] = 1.0 / np.sum(w**2)

# %%

## Band plots: guided prop 1 ##

for lam in true_lams.columns:
    plt.plot(t_system, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(ds_guided_p1_moments["mean"].sel({"lambda": lam}), color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(t_system, 
                     y1=(ds_guided_p1_moments["mean"].sel({"lambda": lam})
                         -2*np.sqrt(ds_guided_p1_moments["var"].sel({"lambda": lam}))), 
                     y2=(ds_guided_p1_moments["mean"].sel({"lambda": lam})
                         +2*np.sqrt(ds_guided_p1_moments["var"].sel({"lambda": lam}))), 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title(f"Guided PF P1 band plot: {lam} | N_RW={N_rws} N={N} DT={DELTA_T}")
    plt.show()


## Band plots: guided prop 2 ##

for lam in true_lams.columns:
    plt.plot(t_system, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(ds_guided_p2_moments["mean"].sel({"lambda": lam}), color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(t_system, 
                     y1=(ds_guided_p2_moments["mean"].sel({"lambda": lam})
                         -2*np.sqrt(ds_guided_p2_moments["var"].sel({"lambda": lam}))), 
                     y2=(ds_guided_p2_moments["mean"].sel({"lambda": lam})
                         +2*np.sqrt(ds_guided_p2_moments["var"].sel({"lambda": lam}))), 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title(f"Guided PF P2 band plot: {lam} | N_RW={N_rws} N={N} DT={DELTA_T}")
    plt.show()

# %%

## ESS over time: guided prop 1 ##

plt.plot(t_system, ess_guided_p1, color="red")
plt.xlabel("t")
plt.ylabel("ESS")
plt.title(f"ESS over time: guided PF P1 | N_RW={N_rws} N={N} DT={DELTA_T}")
plt.show()


## ESS over time: guided prop 2 ##

plt.plot(t_system, ess_guided_p2, color="red")
plt.xlabel("t")
plt.ylabel("ESS")
plt.title(f"ESS over time: guided PF P2 | N_RW={N_rws} N={N} DT={DELTA_T}")
plt.show()

# %%

## Smooth moments: guided prop 1 ##

ds_guided_p1_moments_smooth = ds_guided_p1_moments.copy()
for _ in range(3): # Smooth multiple times
    ds_guided_p1_moments_smooth = ds_guided_p1_moments_smooth.rolling(
            time=11, center=True, min_periods=1).mean()


## Smooth moments: guided prop 2 ##

ds_guided_p2_moments_smooth = ds_guided_p2_moments.copy()
for _ in range(3): # Smooth multiple times
    ds_guided_p2_moments_smooth = ds_guided_p2_moments_smooth.rolling(
            time=11, center=True, min_periods=1).mean()

# %%

## Band plot smoothed moments: guided prop 1 ##

for lam in true_lams.columns:
    plt.plot(t_system, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(ds_guided_p1_moments_smooth["mean"].sel({"lambda": lam}), color="green",
             label="PF smoothed mean", alpha=0.7)
    plt.fill_between(t_system, 
                     y1=(ds_guided_p1_moments_smooth["mean"].sel({"lambda": lam})
                         -2*np.sqrt(ds_guided_p1_moments_smooth["var"].sel({"lambda": lam}))), 
                     y2=(ds_guided_p1_moments_smooth["mean"].sel({"lambda": lam})
                         +2*np.sqrt(ds_guided_p1_moments_smooth["var"].sel({"lambda": lam}))), 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title(f"Guided PF P1 smoothed band plot: {lam} | N_RW={N_rws} N={N} DT={DELTA_T}")
    plt.show()


## Band plot smoothed moments: guided prop 2 ##

for lam in true_lams.columns:
    plt.plot(t_system, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(ds_guided_p2_moments_smooth["mean"].sel({"lambda": lam}), color="green",
             label="PF smoothed mean", alpha=0.7)
    plt.fill_between(t_system, 
                     y1=(ds_guided_p2_moments_smooth["mean"].sel({"lambda": lam})
                         -2*np.sqrt(ds_guided_p2_moments_smooth["var"].sel({"lambda": lam}))), 
                     y2=(ds_guided_p2_moments_smooth["mean"].sel({"lambda": lam})
                         +2*np.sqrt(ds_guided_p2_moments_smooth["var"].sel({"lambda": lam}))), 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title(f"Guided PF P2 smoothed band plot: {lam} | N_RW={N_rws} N={N} DT={DELTA_T}")
    plt.show()

# %%

## Plot histograms/KDEs and box-plots at t ##
# t = 200

## KDEs of all rate parameters ##

# for lam in true_lams.columns:
#     fig, ax = plt.subplots(figsize=(8, 6))
#     # sns.kdeplot(x=da_boot.where(da_boot['lambda'] == lam, drop=True)[t].values.reshape(-1),
#     #            weights=pf_boot.hist.wgts[t].W, ax=ax, fill=True,
#     #            color="skyblue", label="Boot")
#     sns.kdeplot(x=da_guided.where(da_guided['lambda'] == lam, drop=True)[t].values.reshape(-1),
#                 weights=pf_guided.hist.wgts[t].W, ax=ax, fill=True,
#                 color="lightcoral", label="Guided")
#     ax.axvline(x=true_lams.loc[t][lam], color='red', linestyle=':', linewidth=1.5, 
#                label='True state')
#     ax.set_xlabel("Value")
#     ax.set_ylabel("Density")
#     ax.set_title(f"Filtering Dist @ t = {t}: {lam}")
#     ax.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.show()

# %%

## Pairwise Scatter & Contour plots ##

# Bootstrap

# plot_df = (
#     da_boot.sel(time=t)
#     .to_pandas() # index: particle, columns: lambda
#     .reset_index(drop=True)
# )
# sns.pairplot(
#     plot_df,
#     plot_kws={"alpha": 0.5, "s": 15},
#     diag_kind="kde"
# )
# plt.suptitle(f"Pairwise scatter at time t = {t}: Boot", y=1.02)
# plt.show()

# Guided

# plot_df = (
#     da_guided.sel(time=t)
#     .to_pandas() # index: particle, columns: lambda
#     .reset_index(drop=True)
# )
# sns.pairplot(
#     plot_df,
#     plot_kws={"alpha": 0.5, "s": 15},
#     diag_kind="kde"
# )
# plt.suptitle(f"Pairwise scatter at time t = {t}: Guided", y=1.02)
# plt.show()

# %%

## Overlay true values ##

# g = sns.pairplot(plot_df,
#                  plot_kws={"alpha": 0.5, "s": 15},
#                  diag_kind="kde")

# true_vals_t = {
#     true_lams.columns[0]: true_lams.loc[t][true_lams.columns[0]],
#     true_lams.columns[1]: true_lams.loc[t][true_lams.columns[1]],
# }

# for i, y_var in enumerate(g.y_vars):
#     for j, x_var in enumerate(g.x_vars):
#         ax = g.axes[i, j]
#         if i != j:
#             ax.scatter(true_vals_t[x_var], true_vals_t[y_var],
#                        color="red", s=80, zorder=10)

# for i, var in enumerate(g.x_vars):
#     g.axes[i, i].axvline(true_vals_t[var],
#                          color="red", linestyle=":", linewidth=2)

# plt.show()


# %%



