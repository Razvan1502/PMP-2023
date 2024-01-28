import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import pytensor.tensor as pt

""" am luat 1.75/2

1. Generaţi 500 de date dintr-o mixtură de trei distribuţii Gaussiene. În fişierul alăturat aveţi un astfel de
exemplu. (0.5p)

2. Calibraţi pe acest set de date un model de mixtură de distribuţii Gaussiene cu 2, 3, respectiv 4 compo-
nente. (1p)

3. Comparaţi cele 3 modele folosind metodele WAIC şi LOO. Care este concluzia? (0.5p)
"""

def run_model(cluster, mix):
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster)) # prior for mixture weights
        means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), cluster), sigma=10, shape=cluster) # prior for means
        sd = pm.HalfNormal('sd', sigma=10) # prior for standard deviation
        order_means = pm.Potential('order_means', pt.switch(means[1] - means[0] < 0, -np.inf, 0)) # ensure means are ordered
        y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix) # likelihood
        idata = pm.sample(100, target_accept=0.9, random_seed=123, return_inferencedata=True)
    return idata, model

if __name__ == "__main__":
    # ex1
    clusters = 3 # number of clusters
    n_cluster = [200, 150, 150] # number of samples in each cluster
    n_total = sum(n_cluster) # total number of samples
    means = [5, 0, 10] # means of each cluster
    std_devs = [2, 2, 2] # standard deviations of each cluster
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster)) # generate data
    az.plot_kde(np.array(mix))

    # ex2
    clusters_list = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters_list: # run model for each cluster
        idata, model = run_model(cluster, mix)
        idatas.append(idata) # save idata
        models.append(model) # save model

    # ex3
    for i in range(3):
        pm.compute_log_likelihood(idatas[i], model=models[i]) # compute log likelihood for each model

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
