# %%

# -*- coding: utf-8 -*-

""" Simple Exponential Growth 2D States """

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
K = 0.8 # Nudge factor of guided P

def get_obs(t):
    """ Returns true if an observation is aquired at this time. """
    return t > 0 #and t % DELTA_T == 0

class SimpleExpGrowth2D(ssm.StateSpaceModel):
    """
    Simple exponential growth SSM
    """

    def PX0(self):
        return dists.MvNormal(loc=self.mu0, scale=self.sigma0, cov=self.cov0)

    def PX(self, t, xp):
        mut_0 = self.alpha * xp[:,0] - self.beta * xp[:,1]
        mut_1 = self.alpha * xp[:,1] - (1-self.beta) * xp[:,0]
        return dists.MvNormal(loc=np.vstack((mut_0, mut_1)).T,
                              scale=self.sigma, cov=self.cov)

    def PY(self, t, xp, x):
        if get_obs(t):
            return dists.Normal(loc=x[:,0], scale=self.gamma)
        else:
            return dists.FlatNormal(loc=np.zeros(len(x)))

class SimpleExpGrowth_proposal(SimpleExpGrowth2D): # A non-optimal proposal
    """ A non-optimal proposal """
    
    def proposal0(self, data): # t = 0
        if np.isnan(data[0]):
            return self.PX0()
        else:
            return self.PX0() # EDIT if data exists at t = 0
    def proposal(self, t, xp, data): # t >= 1
        if np.isnan(data[t]):
            return self.PX(t, xp)
        else:
            mut_0 = self.alpha * xp[:,0] - self.beta * xp[:,1]
            mut_1 = self.alpha * xp[:,1] - (1-self.beta) * xp[:,0]
            C = np.cov(xp, rowvar=False)
            # Getting error with broadcasting. I'll sort this later
            nudge = K * (data[t] - mut_0) * np.array([C[0,0], C[1,1]])
            new_mut_0 = mut_0 + nudge[0]
            new_mut_1 = mut_1 + nudge[1]
            return dists.MvNormal(loc=np.vstack((new_mut_0, new_mut_1)).T,
                                  scale=self.sigma, cov=self.cov)

class SimpleExpGrowth_proposal_lf(SimpleExpGrowth2D): # A non-optimal proposal
    """ A non-optimal proposal with look forward (for sparse observations) """
    
    def proposal0(self, data): # t = 0
        if np.isnan(data[0]):
            return self.PX0()
        else:
            return self.PX0() # EDIT if data[0] exists
    def proposal(self, t, xp, data): # t >= 1
        if np.isnan(data[t]):
            k = DELTA_T - t % DELTA_T
            prop_ps = []
            for p in xp:
                mut_0 = self.alpha * p[0] - self.beta * p[1]
                mut_1 = self.alpha * p[1] - (1-self.beta) * p[0]
                prop_ps.append([mut_0, mut_1])
            prop_ps_t_0 = np.array([p[0] for p in prop_ps])
            for _ in range(k):
                for i in range(len(prop_ps)):
                    p = prop_ps[i]
                    mut_0 = self.alpha * p[0] - self.beta * p[1]
                    mut_1 = self.alpha * p[1] - (1-self.beta) * p[0]
                    prop_ps[i] = [mut_0, mut_1]
            weights = []
            if not get_obs(t+k): # To check
                raise Exception("t + k is not where data is!")
            for p in prop_ps:
                weights.append(stats.norm.pdf(data[t+k],
                                               loc=p[0],
                                               scale=self.gamma))
            weights = np.array(weights).reshape(len(xp),)
            weights = weights / np.sum(weights) # Normalise weights
            data_t_est = np.average(prop_ps_t_0, weights=weights)
            
            mut_0 = self.alpha * xp[:,0] - self.beta * xp[:,1]
            mut_1 = self.alpha * xp[:,1] - (1-self.beta) * xp[:,0]
            C = np.cov(xp, rowvar=False)
            # Getting error with broadcasting. I'll sort this later
            nudge = K * (data_t_est - mut_0) * np.array([C[0,0], C[1,1]])
            new_mut_0 = mut_0 + nudge[0]
            new_mut_1 = mut_1 + nudge[1]
            return dists.MvNormal(loc=np.vstack((new_mut_0, new_mut_1)).T,
                                  scale=self.sigma, cov=self.cov)
        else:
            mut_0 = self.alpha * xp[:,0] - self.beta * xp[:,1]
            mut_1 = self.alpha * xp[:,1] - (1-self.beta) * xp[:,0]
            C = np.cov(xp, rowvar=False)
            # Getting error with broadcasting. I'll sort this later
            nudge = K * (data[t] - mut_0) * np.array([C[0,0], C[1,1]])
            new_mut_0 = mut_0 + nudge[0]
            new_mut_1 = mut_1 + nudge[1]
            return dists.MvNormal(loc=np.vstack((new_mut_0, new_mut_1)).T,
                                  scale=self.sigma, cov=self.cov)

## Define parameters ##
mu0 = np.array([0, 0]) 
sigma0 = np.array([1, 1])
cov0 = np.array([[1, 0], [0, 1]])
sigma = np.array([0.6, 0.9])
cov = np.array([[1, 0], [0, 1]])
gamma = 0.1 # Observation noise
alpha = 1.05
beta = 0.7

## Define time period ##
T = 4 # t = 0, 1, ..., T
t_obs = [t for t in range(T+1) if get_obs(t)]
t_system = range(T+1)

## Number of particles for PFs ##
N = 10000

seg_ssm = SimpleExpGrowth_proposal(mu0 = mu0, sigma0 = sigma0, cov0 = cov0, 
                          sigma = sigma, cov = cov, 
                          gamma = gamma, 
                          alpha = alpha, beta = beta)
seg_ssm_lf = SimpleExpGrowth_proposal_lf(mu0 = mu0, sigma0 = sigma0, cov0 = cov0, 
                          sigma = sigma, cov = cov, 
                          gamma = gamma, 
                          alpha = alpha, beta = beta)

true_states, data = seg_ssm.simulate(T+1)
true_states_0 = []
true_states_1 = []
for state in true_states:
    true_states_0.append(state[0][0])
    true_states_1.append(state[0][1])
data_clean = [val for val in data if not np.isnan(val)] # For plotting

# %%

# True states 0
plt.plot(t_system, true_states_0, label="true state 0", color='red')
plt.plot(t_obs, data_clean, label="observation", color='blue', marker="o",
         alpha=0.7)
plt.xlabel("t")
plt.ylabel("Value")
plt.title("True states 0 + Observations")
plt.legend()
plt.grid(True)
plt.show()

# True states 1
plt.plot(t_system, true_states_1, label="true state 1", color='purple')
plt.xlabel("t")
plt.ylabel("Value")
plt.title("True states 1")
plt.legend()
plt.grid(True)
plt.show()

# %%

## Bootstrap PF ##

fk_boot = ssm.Bootstrap(ssm=seg_ssm, data=data)
pf_boot = particles.SMC(fk=fk_boot, N=N, resampling='stratified', 
                        store_history=True, collect=[Moments()])
pf_boot.run()

# %%

## Guided PF no LF ##

fk_guided = ssm.GuidedPF(ssm=seg_ssm, data=data)
pf_guided = particles.SMC(fk=fk_guided, N=N, resampling='stratified', 
                          store_history=True, collect=[Moments()])
pf_guided.run()

# %%

## Guided PF with LF ##

fk_guided_lf = ssm.GuidedPF(ssm=seg_ssm_lf, data=data)
pf_guided_lf = particles.SMC(fk=fk_guided_lf, N=N, resampling='stratified', 
                             store_history=True, collect=[Moments()])
pf_guided_lf.run()

# %%

#### Filtering band plots ####

## Bootstrap ##

means_boot =  np.stack([m['mean'] for m in pf_boot.summaries.moments])
vars_boot = np.stack([m['var'] for m in pf_boot.summaries.moments])

plt.plot(t_system, true_states_0, label="true state 0", color='red', alpha=0.7)
plt.plot(means_boot[:,0], color="green", label="PF mean", alpha=0.7)
plt.plot(t_obs, data_clean, label="observation", color='blue', alpha=0.3,
         marker='o')
plt.fill_between(t_system, 
                 y1=means_boot[:,0]-2*np.sqrt(vars_boot[:,0]), 
                 y2=means_boot[:,0]+2*np.sqrt(vars_boot[:,0]), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Bootstrap PF band plot: states 0")
plt.show()

plt.plot(t_system, true_states_1, label="true state 1", color='red', alpha=0.7)
plt.plot(means_boot[:,1], color="green", label="PF mean", alpha=0.7)
plt.fill_between(t_system, 
                 y1=means_boot[:,1]-2*np.sqrt(vars_boot[:,1]), 
                 y2=means_boot[:,1]+2*np.sqrt(vars_boot[:,1]), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Bootstrap PF band plot: states 1")
plt.show()

# %%
## Guided no LF ##

means_guided =  np.stack([m['mean'] for m in pf_guided.summaries.moments])
vars_guided = np.stack([m['var'] for m in pf_guided.summaries.moments])

plt.plot(t_system, true_states_0, label="true state 0", color='red', alpha=0.7)
plt.plot(means_guided[:,0], color="green", label="PF mean", alpha=0.7)
plt.plot(t_obs, data_clean, label="observation", color='blue', alpha=0.3,
         marker='o')
plt.fill_between(t_system, 
                 y1=means_guided[:,0]-2*np.sqrt(vars_guided[:,0]), 
                 y2=means_guided[:,0]+2*np.sqrt(vars_guided[:,0]), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Guided PF band plot: states 0")
plt.show()

plt.plot(t_system, true_states_1, label="true state 1", color='red', alpha=0.7)
plt.plot(means_guided[:,1], color="green", label="PF mean", alpha=0.7)
plt.fill_between(t_system, 
                 y1=means_guided[:,1]-2*np.sqrt(vars_guided[:,1]), 
                 y2=means_guided[:,1]+2*np.sqrt(vars_guided[:,1]), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Guided PF band plot: states 1")
plt.show()

# %%
## Guided with LF ##

means_guided_lf =  np.stack([m['mean'] for m in pf_guided_lf.summaries.moments])
vars_guided_lf = np.stack([m['var'] for m in pf_guided_lf.summaries.moments])

plt.plot(t_system, true_states_0, label="true state 0", color='red', alpha=0.7)
plt.plot(means_guided_lf[:,0], color="green", label="PF mean", alpha=0.7)
plt.plot(t_obs, data_clean, label="observation", color='blue', alpha=0.3,
         marker='o')
plt.fill_between(t_system, 
                 y1=means_guided_lf[:,0]-2*np.sqrt(vars_guided_lf[:,0]), 
                 y2=means_guided_lf[:,0]+2*np.sqrt(vars_guided_lf[:,0]), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Guided-LF PF band plot: states 0")
plt.show()

plt.plot(t_system, true_states_1, label="true state 1", color='red', alpha=0.7)
plt.plot(means_guided_lf[:,1], color="green", label="PF mean", alpha=0.7)
plt.fill_between(t_system, 
                 y1=means_guided_lf[:,1]-2*np.sqrt(vars_guided_lf[:,1]), 
                 y2=means_guided_lf[:,1]+2*np.sqrt(vars_guided_lf[:,1]), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Guided-LF PF band plot: states 1")
plt.show()

# %%

## Plot histograms/KDEs and box-plots at t = n ##
n = 1

## Box plots ##

# Boxplots aren't correct because weights aren't included. See KDEs
plt.boxplot([pf_boot.hist.X[n][:, 0], pf_guided.hist.X[n][:, 0], 
             pf_guided_lf.hist.X[n][:, 0]],
            tick_labels=["Boot", "Guided", "Guided-LF"])
plt.scatter([1, 2, 3], [true_states_0[n], true_states_0[n], true_states_0[n]],
            color='red', marker='x', s=100, label='True state')
plt.title('Filtering Dists @ t=n: state 0')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.boxplot([pf_boot.hist.X[n][:, 1], pf_guided.hist.X[n][:, 1], 
             pf_guided_lf.hist.X[n][:, 1]],
            tick_labels=["Boot", "Guided", "Guided-LF"])
plt.scatter([1, 2, 3], [true_states_1[n], true_states_1[n], true_states_1[n]],
            color='purple', marker='x', s=100, label='True state')
plt.title('Filtering Dists @ t=n: state 1')
plt.ylabel('Value')
plt.legend()
plt.show()

# %%

## KDEs ##

# Add theoretical filtering distribution: t = 1 (Need data at t=1)
A = np.array([[alpha, -beta], [-(1-beta), alpha]])
sigma0_2 = np.array([[1, 0], [0, 1]])
sigma_2 = np.array([[0.6**2, 0], [0, 0.9**2]])
Q1 = A @ sigma0_2 @ A.T + sigma_2
H = np.array([[1, 0]])
KC = Q1 @ H.T @ np.linalg.inv( H @ Q1 @ H.T + gamma ** 2 )
m1 = A @ mu0.reshape(2, 1)
filt_mean = m1 + KC @ (data[1] - H @ m1)
filt_var = (np.identity(2) - KC @ H) @ Q1

# %%

x_grid_0 = np.linspace(-1.4, -0.4, 300)
filt_pdf_vals_0 = stats.norm.pdf(x_grid_0, filt_mean[0], np.sqrt(filt_var[0, 0]))

x_grid_1 = np.linspace(-4, 5, 300)
filt_pdf_vals_1 = stats.norm.pdf(x_grid_1, filt_mean[1], np.sqrt(filt_var[1, 1]))

# %%
fig, ax = plt.subplots(figsize=(8, 6))
plt.xlim(-1.4, -0.4)
sns.kdeplot(x=pf_boot.hist.X[n][:, 0], 
            weights=pf_boot.hist.wgts[n].W, ax=ax, fill=True,
            color="skyblue", label="Boot")
sns.kdeplot(x=pf_guided.hist.X[n][:, 0],
            weights=pf_guided.hist.wgts[n].W, ax=ax, fill=True,
            color="lightcoral", label="Guided")
sns.kdeplot(x=pf_guided_lf.hist.X[n][:, 0],
            weights=pf_guided_lf.hist.wgts[n].W, ax=ax, fill=True,
            color="gold", label="Guided-LF")
ax.plot(x_grid_0, filt_pdf_vals_0, label="True Filtering PDF", linestyle="--")
ax.axvline(x=true_states_0[n], color='red', linestyle=':', linewidth=1.5, 
           label='True state')
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Filtering Dists @ t=n: state 0")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(x=pf_boot.hist.X[n][:, 1],
            weights=pf_boot.hist.wgts[n].W, ax=ax, fill=True,
            color="skyblue", label="Boot")
sns.kdeplot(x=pf_guided.hist.X[n][:, 1],
            weights=pf_guided.hist.wgts[n].W, ax=ax, fill=True,
            color="lightcoral", label="Guided")
sns.kdeplot(x=pf_guided_lf.hist.X[n][:, 1],
            weights=pf_guided_lf.hist.wgts[n].W, ax=ax, fill=True,
            color="gold", label="Guided-LF")
ax.plot(x_grid_1, filt_pdf_vals_1, label="True Filtering PDF", linestyle="--")
ax.axvline(x=true_states_1[n], color='purple', linestyle=':', linewidth=1.5, 
           label='True state')
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Filtering Dists @ t=n: state 1")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
