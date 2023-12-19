import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import pytensor.tensor as pt

def run_model(cluster, mix):
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), cluster), sigma=10, shape=cluster)
        sd = pm.HalfNormal('sd', sigma=10)
        order_means = pm.Potential('order_means', pt.switch(means[1] - means[0] < 0, -np.inf, 0))
        y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
        idata = pm.sample(100, target_accept=0.9, random_seed=123, return_inferencedata=True)
    return idata, model

if __name__ == "__main__":
    # ex1
    clusters = 3
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 0, 10]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))

    # ex2
    clusters_list = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters_list:
        idata, model = run_model(cluster, mix)
        idatas.append(idata)
        models.append(model)

    # ex3
    for i in range(3):
        pm.compute_log_likelihood(idatas[i], model=models[i])

    # compare results using WAIC for models 2, 3, and 4
    cmp_df_waic = az.compare({'model_2': idatas[0], 'model_3': idatas[1], 'model_4': idatas[2]},
                             method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(cmp_df_waic)
    plt.show()

    # compare results using LOO for models 2, 3, and 4
    cmp_df_loo = az.compare({'model_2': idatas[0], 'model_3': idatas[1], 'model_4': idatas[2]},
                            method='BB-pseudo-BMA', ic="loo", scale="deviance")
    az.plot_compare(cmp_df_loo)
    plt.show()
