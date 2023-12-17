import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

dummy_data = np.loadtxt('./data/dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 3
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

if __name__ == '__main__':
    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p_sd_10:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p_sd_10 = pm.sample(2000, return_inferencedata=True)

    
    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    α_p_post = idata_p_sd_10.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p_sd_10.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

    pm.compute_log_likelihood(idata_l, model=model_l)
    waic_l = az.waic(idata_l, scale="deviance")
    print(waic_l)
    az.plot_compare(waic_l)

    pm.compute_log_likelihood(idata_p_sd_10, model=model_p_sd_10)
    cmp_df = az.compare({'model_l': idata_l, 'model_p': idata_p_sd_10},
                        method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(cmp_df)
    az.plot_compare(cmp_df)


