import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal, stats
import sklearn as sk
from sklearn import metrics

import seaborn as sns
#%%
T = 1 + np.arange(100)
#%% md
## A 2-D Gaussian Process
#%%
# MEAN

# STATIONARY
# mu = np.zeros_like(T)

# NON-STATIONARY
# mu = np.linspace(0, 2*np.pi, len(T)) #
# mu = 10*np.sin(np.linspace(0, 2*np.pi, len(T)))

# RANDOM WALK
mu = np.cumsum(np.random.randn(len(T)))
#%%
# COVARIANCE

## White Noise
sigma = np.eye(len(T))

## Growing Noise
# sigma = np.diag(np.logspace(-2, 0, len(T)))

## NON-DIAGONAL COVARIANCES
## Covariance in local neighborhood
# sigma = sk.metrics.pairwise.rbf_kernel(
#     np.expand_dims(T.astype(float)*.1, -1))

## Covariance between distant points
# sigma += np.fliplr(sigma)
# sigma /= 10
# sigma +=.8
#%%
sns.heatmap(sigma)
#%%
process = sp.stats.multivariate_normal(mean=mu,
                                       cov=sigma,
                                       allow_singular=True)
#%%
df = pd.DataFrame()
for i in range(3):
    temp = pd.DataFrame(
        data=process.rvs(),
        index=pd.Index(T, name='T'),
        columns=pd.Index(['X'], name="outcomes")
    )
    temp['mean'] = mu
    temp['run'] = i
    df = pd.concat((df, temp))

df.reset_index(inplace=True)

# Modify df to have runs as variables (i.e. columns) and outcomes as index
df = df.pivot_table(index=['T', 'mean'], columns='run', values='X')
df = df.pivot_table(index='T', columns=['run'], values=['X', 'mean'])

print(df.head(10))