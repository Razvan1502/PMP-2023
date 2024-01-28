import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

def run_pymc_model():
    Prices = pd.read_csv('Prices.csv')
    x_1 = Prices['Speed'].values
    x_2 = np.log(Prices['HardDrive'].values)
    y = Prices['Price'].values
    X = np.column_stack((x_1,x_2))
    X_mean = X.mean(axis=0, keepdims=True)
    #pentru a ne face o idee asupra mediilor si dev. standard:
    print(X_mean)
    print(y.mean())
    print(X.std(axis=0, keepdims=True))
    print(y.std())

# si o idee despre date:
    def scatter_plot(x, y):
        plt.figure(figsize=(15, 5))
        for idx, x_i in enumerate(x.T):
            plt.subplot(1, 3, idx+1)
            plt.scatter(x_i, y)
            plt.xlabel(f'x_{idx+1}')
            plt.ylabel(f'y', rotation=0)

        plt.subplot(1, 3, idx+2)
        plt.scatter(x[:, 0], x[:, 1])
        plt.xlabel(f'x_{idx}')
        plt.ylabel(f'x_{idx+1}', rotation=0)
    scatter_plot(X, y)

    with pm.Model() as model_mlr:
        α = pm.Normal('α', mu=0, sigma=1000)
        # am luat sigma=1000 deoarece nu am standardizat datele, iar dev. standard pentru y este f. mare
        β = pm.Normal('β', mu=0, sigma=1000, shape=2)
        ϵ = pm.HalfCauchy('ϵ', 5000)
        ν = pm.Exponential('ν', 1 / 30)
        X_shared = pm.MutableData('x_shared', X)  # pentru pct. 5
        μ = pm.Deterministic('μ', α + pm.math.dot(X_shared, β))

        y_pred = pm.StudentT('y_pred', mu=μ, sigma=ϵ, nu=ν, observed=y)

        idata_mlr = pm.sample(1250, return_inferencedata=True)

    az.plot_forest(idata_mlr, hdi_prob=0.95, var_names=['β'])
    az.summary(idata_mlr, hdi_prob=0.95, var_names=['β'])

    posterior_g = idata_mlr.posterior.stack(samples={"chain", "draw"}) #avem 5000 de extrageri in esantion (nr. draws x nr. chains)
    mu = posterior_g['α']+33*posterior_g['β'][0]+np.log(540)*posterior_g['β'][1]
    az.plot_posterior(mu.values,hdi_prob=0.9)

    pm.set_data({"x_shared":[[33,np.log(540)]]}, model=model_mlr)
    ppc = pm.sample_posterior_predictive(idata_mlr, model=model_mlr)
    y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=("chain", "draw")).values
    az.plot_posterior(y_ppc,hdi_prob=0.9)

if __name__ == '__main__':
    run_pymc_model()