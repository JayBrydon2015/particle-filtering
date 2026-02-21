# %%

# -*- coding: utf-8 -*-

""" Implements a LV SSM and runs a particle filter for state estimation """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Particles package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments

## Constants ##
DELTA_T = 2 # Data collected every DELTA_T-th time.
K = 0.8 # Nudge factor of guided PF

def get_obs(t):
    """ Returns true if an observation is aquired at this time. """
    return t > 0 and t % DELTA_T == 0

class LotkaVolterra(ssm.StateSpaceModel):
    """
    Predator-Prey Model

    Parameters:
        alpha, beta, gamma, delta are the model rate parameters.
        h is the constant change in time.
        n0 is the mean initial predator population; n1 the mean initial prey population.
        sigmaPrime and tauPrime are the variances of the initial log-predator and log-prey populations.
        sigma and tau are the variances of the transition densities.
        theta is the probability of observing a particular prey.
        Note: alpha, beta, gamma, delta > 0. n0, n1 > 0.
    """

    def PX0(self):
        d = dists.MvNormal(loc=np.array([np.log(self.n0), np.log(self.n1)]),
                           scale=np.array([(self.sigma0), (self.tau0)])) 
        return d

    def PX(self, t, xp):
        mu0 = self.h * (self.beta * np.exp(xp[:,1]) - self.alpha) + xp[:,0]
        mu1 = self.h * (self.gamma - self.delta * np.exp(xp[:,0])) + xp[:,1]
        d = dists.MvNormal(loc=np.vstack((mu0,mu1)).T,
                           scale=np.array([self.sigma, self.tau]))
        return d

    def PY(self, t, xp, x):
        if get_obs(t):
            return dists.Binomial(n=np.rint(np.exp(x[:,0])).astype(int),
                                  p=self.theta)
        else:
            return dists.FlatNormal(loc=np.zeros(len(x)))

class LotkaVolterra_proposal(LotkaVolterra): # No look-forward
    def proposal0(self, data): # t = 0
        if np.isnan(data[0]):
            return self.PX0()
        else:
            return self.PX0() # EDIT
    def proposal(self, t, xp, data): # t >= 1
        if np.isnan(data[t]):
            return self.PX(t, xp)
        else:
            mu0 = self.h * (self.beta * np.exp(xp[:,1]) - self.alpha) + xp[:,0]
            mu1 = self.h * (self.gamma - self.delta * np.exp(xp[:,0])) + xp[:,1]
            nudge = K * (np.log(data[t] / self.theta) - mu0)
            new_mu0 = mu0 + nudge
            return dists.MvNormal(loc=np.vstack((new_mu0,mu1)).T,
                              scale=np.array([self.sigma, self.tau]))

class LotkaVolterra_proposal_lf(LotkaVolterra):
    def proposal0(self, data): # t = 0
        if np.isnan(data[0]):
            return self.PX0()
        else:
            return self.PX0() # EDIT if data exists at t = 0
    def proposal(self, t, xp, data): # t >= 1
        if np.isnan(data[t]):
            k = DELTA_T - t % DELTA_T
            prop_ps = []
            for p in xp:
                mu0 = self.h * (self.beta * np.exp(p[1]) - self.alpha) + p[0]
                mu1 = self.h * (self.gamma - self.delta * np.exp(p[0])) + p[1]
                prop_ps.append(np.array([mu0, mu1]))
            prop_ps_t_pred = np.array([p[0] for p in prop_ps])
            for _ in range(k):
                for i in range(len(prop_ps)):
                    p = prop_ps[i]
                    next_mu0 = self.h * (self.beta * np.exp(p[1]) - self.alpha) + p[0]
                    next_mu1 = self.h * (self.gamma - self.delta * np.exp(p[0])) + p[1]
                    prop_ps[i] = np.array([next_mu0, next_mu1])
            weights = []
            if not get_obs(t+k): # To check
                raise Exception("t + k is not where data is!")
            for p in prop_ps:
                # Issue: weights 0 unless (1) == (2), if self.theta == 0
                weights.append(stats.binom.pmf(data[t+k], # (1)
                                               np.rint(np.exp(p[0])).astype(int), # (2)
                                               self.theta))
            weights = np.array(weights).reshape(len(xp),)
            weights = weights / np.sum(weights) # Normalise weights
            data_t_est = np.sum(prop_ps_t_pred * weights)
            
            mu0 = self.h * (self.beta * np.exp(xp[:,1]) - self.alpha) + xp[:,0]
            mu1 = self.h * (self.gamma - self.delta * np.exp(xp[:,0])) + xp[:,1]
            nudge = K * (data_t_est - mu0)
            new_mu0 = mu0 + nudge
            return dists.MvNormal(loc=np.vstack((new_mu0,mu1)).T,
                              scale=np.array([self.sigma, self.tau]))
        else:
            mu0 = self.h * (self.beta * np.exp(xp[:,1]) - self.alpha) + xp[:,0]
            mu1 = self.h * (self.gamma - self.delta * np.exp(xp[:,0])) + xp[:,1]
            nudge = K * (np.log(data[t] / self.theta) - mu0)
            new_mu0 = mu0 + nudge
            return dists.MvNormal(loc=np.vstack((new_mu0,mu1)).T,
                              scale=np.array([self.sigma, self.tau]))

## Define parameters ##
alpha = 0.5; beta = 0.02; gamma = 0.8; delta = 0.01
h = 0.002
n0 = 20; n1 = 40
sigma0 = 0.5; tau0 = 0.6
sigma = 0.02; tau = 0.01 # Need to go smaller for smaller h
theta = 0.9

## Define time period ##
T = 10 # t = 0, 1, ..., T
t_obs = [t for t in range(T+1) if get_obs(t)]
t_system = range(T+1)

## Number of particles ##
N = 200

## Run SSM simulation ##
lv_ssm = LotkaVolterra_proposal(alpha=alpha, beta=beta, gamma=gamma,
                                delta=delta, h=h, n0=n0, n1=n1,
                                sigma0=sigma0, tau0=tau0,
                                sigma=sigma, tau=tau, theta=theta)

lv_ssm_lf = LotkaVolterra_proposal_lf(alpha=alpha, beta=beta, gamma=gamma,
                                delta=delta, h=h, n0=n0, n1=n1,
                                sigma0=sigma0, tau0=tau0,
                                sigma=sigma, tau=tau, theta=theta)

true_states, data = lv_ssm.simulate(T+1)
data_clean = [val for val in data if not np.isnan(val)] # For plotting
pred_vals = []
prey_vals = []
for i in range(T+1): # pop or log-pop
        pred, prey = true_states[i][0]
        pred_vals.append(pred)
        prey_vals.append(prey)

# %%
## Plot simulation results ##

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,6))

# Top plot: observed predator population
# ax1.plot(t_obs, data_clean, '.-', label="pred obs") # If obs are sparse
ax1.plot(t_obs, data_clean, label="pred obs")
ax1.set_ylabel("population")
ax1.legend()
ax1.grid(True)

# Bottom plot: true predator + prey (log-)populations
ax2.plot(t_system, pred_vals, label="Predator", color="red")
ax2.plot(t_system, prey_vals, label="Prey", color="blue")
# Add vertical lines at observation times onto bottom plot
#for t in t_obs:
#    ax2.axvline(x=t, color="green", linestyle="--", alpha=0.2)
ax2.set_xlabel("t")
ax2.set_ylabel("log-population")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()

# %%

## Bootstrap Particle Filter ##

fk_boot = ssm.Bootstrap(ssm=lv_ssm, data=data)
pf_boot = particles.SMC(fk=fk_boot, N=N, resampling='stratified', 
                        store_history=True, collect=[Moments()])
pf_boot.run()

# %%

## Guided Particle Filter ##

fk_guided = ssm.GuidedPF(ssm=lv_ssm, data=data)
pf_guided = particles.SMC(fk=fk_guided, N=N, resampling='stratified', 
                          store_history=True, collect=[Moments()])
pf_guided.run()

# %%

## Guided-LF Particle Filter ##

fk_guided_lf = ssm.GuidedPF(ssm=lv_ssm_lf, data=data)
pf_guided_lf = particles.SMC(fk=fk_guided_lf, N=N, resampling='stratified', 
                          store_history=True, collect=[Moments()])
pf_guided_lf.run()

# %%

## Plot histograms, box-plots, and KDEs at t = n ##
n = 1

## Box plots ##

# Predator
plt.boxplot([pf_boot.hist.X[n][:, 0], pf_guided.hist.X[n][:, 0], 
             pf_guided_lf.hist.X[n][:, 0]],
            tick_labels=["Boot", "Guided", "Guided-LF"])
plt.scatter([1, 2, 3], [pred_vals[n], pred_vals[n], pred_vals[n]],
            color='red', marker='x', s=100, label='True log-pop')
plt.title('Predator Log-Population Filtering Dists @ t=n')
plt.ylabel('Log-pop')
plt.legend()
plt.show()

# Prey
plt.boxplot([pf_boot.hist.X[n][:, 1], pf_guided.hist.X[n][:, 1], 
             pf_guided_lf.hist.X[n][:, 1]],
            tick_labels=["Boot", "Guided", "Guided-LF"])
plt.scatter([1, 2, 3], [prey_vals[n], prey_vals[n], prey_vals[n]],
            color='red', marker='x', s=100, label='True log-pop')
plt.title('Prey Log-Population Filtering Dists @ t=n')
plt.ylabel('Log-pop')
plt.legend()
plt.show()

## KDEs ##

# Predator
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(x=pf_boot.hist.X[n][:, 0], 
            weights=pf_boot.hist.wgts[n].W[:, 0], ax=ax, fill=True,
            color="skyblue", label="Boot")
sns.kdeplot(x=pf_guided.hist.X[n][:, 0],
            weights=pf_guided.hist.wgts[n].W[:, 0], ax=ax, fill=True,
            color="lightcoral", label="Guided")
sns.kdeplot(x=pf_guided_lf.hist.X[n][:, 0],
            weights=pf_guided_lf.hist.wgts[n].W[:, 0], ax=ax, fill=True,
            color="gold", label="Guided-LF")
ax.axvline(x=pred_vals[n], color='red', linestyle=':', linewidth=1.5, 
           label='True log-pop')
ax.set_xlabel("Log-pop")
ax.set_ylabel("Density")
ax.set_title("Predator Log-Population Filtering Dists @ t=n")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Prey
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(x=pf_boot.hist.X[n][:, 1],
            weights=pf_boot.hist.wgts[n].W[:, 1], ax=ax, fill=True,
            color="skyblue", label="Boot")
sns.kdeplot(x=pf_guided.hist.X[n][:, 1],
            weights=pf_guided.hist.wgts[n].W[:, 1], ax=ax, fill=True,
            color="lightcoral", label="Guided")
sns.kdeplot(x=pf_guided_lf.hist.X[n][:, 1],
            weights=pf_guided_lf.hist.wgts[n].W[:, 1], ax=ax, fill=True,
            color="gold", label="Guided-LF")
ax.axvline(x=prey_vals[n], color='red', linestyle=':', linewidth=1.5, 
           label='True log-pop')
ax.set_xlabel("Log-pop")
ax.set_ylabel("Density")
ax.set_title("Prey Log-Population Filtering Dists @ t=n")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# =============================================================================
# ## Histograms ##
# 
# # Predator
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(pf_boot.hist.X[n][:, 0], bins=30, alpha=0.5, label='Boot',
#         color='skyblue', density=True)
# ax.hist(pf_guided.hist.X[n][:, 0], bins=30, alpha=0.5, label='Guided', 
#         color='salmon', density=True)
# ax.set_title("Predator Log-Population Filtering Dists @ t=n")
# ax.set_xlabel("Log-pop")
# ax.set_ylabel('Density')
# ax.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
# 
# # Prey
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(pf_boot.hist.X[n][:, 1], bins=30, alpha=0.5, label='Boot',
#         color='skyblue', density=True)
# ax.hist(pf_guided.hist.X[n][:, 1], bins=30, alpha=0.5, label='Guided', 
#         color='salmon', density=True)
# ax.set_title("Prey Log-Population Filtering Dists @ t=n")
# ax.set_xlabel("Log-pop")
# ax.set_ylabel('Density')
# ax.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
# =============================================================================

# %%
## Filtering interval plots ##

# Plot up to t = t_plot_max (instead of T)
t_plot_max = T
t_plot_max += 1
t_plot_values = range(t_plot_max)

# For plotting data
data_clean_log = np.log([dp / theta for dp in data_clean])

## Bootstrap ##

means_boot =  np.stack([m['mean'] for m in pf_boot.summaries.moments])
vars_boot = np.stack([m['var'] for m in pf_boot.summaries.moments])

plt.plot(t_plot_values, pred_vals[0:t_plot_max], label="Predator", color="red")
plt.plot(t_obs, data_clean_log, label="observation", color='purple', alpha=0.3)
plt.plot(means_boot[0:t_plot_max,0], color="orange", label="PF means")
plt.fill_between(t_plot_values, 
                 y1=means_boot[0:t_plot_max,0]-2*np.sqrt(vars_boot[0:t_plot_max,0]), 
                 y2=means_boot[0:t_plot_max,0]+2*np.sqrt(vars_boot[0:t_plot_max,0]), 
                 color="orange", alpha=0.3)
plt.ylabel("log-population")
plt.xlabel("t")
plt.title("Bootstrap")
plt.legend()
#plt.ylim(3.7, 4.2)
plt.grid(True)
plt.show()

plt.plot(t_plot_values, prey_vals[0:t_plot_max], label="Prey", color="blue")
plt.plot(means_boot[0:t_plot_max,1], color="skyblue", label="PF means")
plt.fill_between(t_plot_values, 
                 y1=means_boot[0:t_plot_max,1]-2*np.sqrt(vars_boot[0:t_plot_max,1]), 
                 y2=means_boot[0:t_plot_max,1]+2*np.sqrt(vars_boot[0:t_plot_max,1]), 
                 color="skyblue", alpha=0.3)
plt.xlabel("t")
plt.ylabel("log-population")
plt.title("Bootstrap")
plt.legend()
plt.grid(True)
plt.show()

# %%
## Guided ##

means_guided =  np.stack([m['mean'] for m in pf_guided.summaries.moments])
vars_guided = np.stack([m['var'] for m in pf_guided.summaries.moments])

plt.plot(t_plot_values, pred_vals[0:t_plot_max], label="Predator", color="red")
plt.plot(t_obs, data_clean_log, label="observation", color='purple', alpha=0.3)
plt.plot(means_guided[0:t_plot_max,0], color="orange", label="PF means")
plt.fill_between(t_plot_values, 
                 y1=means_guided[0:t_plot_max,0]-2*np.sqrt(vars_guided[0:t_plot_max,0]), 
                 y2=means_guided[0:t_plot_max,0]+2*np.sqrt(vars_guided[0:t_plot_max,0]), 
                 color="orange", alpha=0.3)
plt.ylabel("log-population")
plt.xlabel("t")
plt.title("Guided")
plt.legend()
#plt.ylim(3.7, 4.2)
plt.grid(True)
plt.show()

plt.plot(t_plot_values, prey_vals[0:t_plot_max], label="Prey", color="blue")
plt.plot(means_guided[0:t_plot_max,1], color="skyblue", label="PF means")
plt.fill_between(t_plot_values, 
                 y1=means_guided[0:t_plot_max,1]-2*np.sqrt(vars_guided[0:t_plot_max,1]), 
                 y2=means_guided[0:t_plot_max,1]+2*np.sqrt(vars_guided[0:t_plot_max,1]), 
                 color="skyblue", alpha=0.3)
plt.xlabel("t")
plt.ylabel("log-population")
plt.title("Guided")
plt.legend()
plt.grid(True)
plt.show()

# %%
## Guided-LF ##

means_guided_lf =  np.stack([m['mean'] for m in pf_guided_lf.summaries.moments])
vars_guided_lf = np.stack([m['var'] for m in pf_guided_lf.summaries.moments])

plt.plot(t_plot_values, pred_vals[0:t_plot_max], label="Predator", color="red")
plt.plot(t_obs, data_clean_log, label="observation", color='purple', alpha=0.3)
plt.plot(means_guided_lf[0:t_plot_max,0], color="orange", label="PF means")
plt.fill_between(t_plot_values, 
                 y1=means_guided_lf[0:t_plot_max,0]-2*np.sqrt(vars_guided_lf[0:t_plot_max,0]), 
                 y2=means_guided_lf[0:t_plot_max,0]+2*np.sqrt(vars_guided_lf[0:t_plot_max,0]), 
                 color="orange", alpha=0.3)
plt.ylabel("log-population")
plt.xlabel("t")
plt.title("Guided-LF")
plt.legend()
#plt.ylim(3.7, 4.2)
plt.grid(True)
plt.show()

plt.plot(t_plot_values, prey_vals[0:t_plot_max], label="Prey", color="blue")
plt.plot(means_guided_lf[0:t_plot_max,1], color="skyblue", label="PF means")
plt.fill_between(t_plot_values, 
                 y1=means_guided_lf[0:t_plot_max,1]-2*np.sqrt(vars_guided_lf[0:t_plot_max,1]), 
                 y2=means_guided_lf[0:t_plot_max,1]+2*np.sqrt(vars_guided_lf[0:t_plot_max,1]), 
                 color="skyblue", alpha=0.3)
plt.xlabel("t")
plt.ylabel("log-population")
plt.title("Guided-LF")
plt.legend()
plt.grid(True)
plt.show()


