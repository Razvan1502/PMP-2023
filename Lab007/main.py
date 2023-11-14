import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az


df = pd.read_csv('auto-mpg.csv')

data=df[['mpg', 'horsepower']]

data=data.drop(data[data.horsepower == '?'].index) #elimina val care contin ?
print(data)


y = data['mpg'].values
x = data['horsepower'].values

y=np.array(y,dtype=np.float64)
x=np.array(x,dtype=np.float64)

#a
plt.scatter(x, y)
plt.title('Relatia dintre horsepower si mpg')
plt.xlabel('mpg')
plt.ylabel('horsepower', rotation=0)
plt.show()

#b
with pm.Model() as model_g:
    alpha = pm.Normal('alpha', mu=y.mean(), sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    mu = pm.Deterministic('mu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y)
    idata_g = pm.sample(500, return_inferencedata=True)



#c
plt.plot(x, y, 'C0.')
posterior_g = idata_g.posterior.stack(samples={"chain", "draw"})
alpha_m = posterior_g['alpha'].mean().item()
beta_m = posterior_g['beta'].mean().item()
draws = range(0, posterior_g.samples.size, 10)

plt.plot(x, posterior_g['alpha'][draws].values
         + posterior_g['beta'][draws].values * x[:,None], c='gray', alpha=0.5)

plt.plot(x, alpha_m + beta_m * x, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

plt.show()

#d
ppc = pm.sample_posterior_predictive(idata_g, model=model_g)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k',
               label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')

az.plot_hdi(x, ppc.posterior_predictive['y_pred'], hdi_prob=0.95, color='gray')
az.plot_hdi(x, ppc.posterior_predictive['y_pred'], color='gray')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.show()

#Putem spune ca CP si mpg sunt invers proportionale