
import math

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt



raw_data = pd.read_csv('./Prices.csv')
data = raw_data[['Price', 'Speed', 'HardDrive', 'Ram', 'Premium']]
price=data['Price'].values
x1 = data['Speed'].values
x2 = data['HardDrive'].values

price=np.array(price,dtype=np.float64)
x1 = np.array(x1, dtype=np.float64)
x2 = np.array(x2, dtype=np.float64)
x2 = np.log(data['HardDrive'])



with pm.Model() as model_g:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=1)
    beta2 = pm.Normal('beta2', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    mu = pm.Deterministic('mu', alpha + beta1 * x1 + beta2 * x2 )
    price_pred = pm.Normal('price_pred', mu=mu, sigma=eps, observed=price)
    idata_g = pm.sample(50, tune=50,cores=1, return_inferencedata=True)

az.plot_posterior(idata_g, var_names='beta1', hdi_prob=0.95)
plt.show()


az.plot_posterior(idata_g, var_names='beta2', hdi_prob=0.95)
plt.show()

mean_beta1 = idata_g.posterior['beta1'].mean().item()
mean_beta2 = idata_g.posterior['beta2'].mean().item()

print(f"beta1: {mean_beta1}")
print(f"beta2: {mean_beta2}")




# Media lui beta1 este mai mare, astfel Speed va avea o influenta mai mare fata de price in comparatie cu HardDrive

