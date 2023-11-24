import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1
    mu = 10  # media
    sigma = 2  # dev standard

    # timpi medii de asteptare
    timpi_asteptare = np.random.normal(mu, sigma, 100)

    print(timpi_asteptare)

    # 2 modelul in PyMC
    with pm.Model() as model:
        # distributia a priori pt timpul mediu de asteptare
        mu_prior = pm.Normal('mu_prior', mu=10, sigma=5) #am ales aceasta distrib deoarece este una comuna pentru variabile continue
        sigma_prior = pm.HalfNormal('sigma_prior', sigma=5) #aceasta distributie este folosita pentru parametrii de scara deoarece restrange parametrul la val pozitive,avando gama larga de val. posibile

        # distributia a priori a timpului de așteptare
        timp_asteptare = pm.Normal('timp_asteptare', mu=mu_prior, sigma=sigma_prior, observed=timpi_asteptare) # timpul de asteptare este distribuit normal in jurul mediei(mu_prior) cu o deviatie standard (sigma_prior).

    with model:
        trace = pm.sample(2000, tune=1000)

    # afisez rezultatele
    pm.plot_posterior(trace, var_names=['mu_prior', 'sigma_prior'], figsize=(12, 6))
    plt.show()

    #distrib a posteriori mu
    posterior_mu = trace['mu_prior']

    # distributia a posteriori pentru mu
    plt.figure(figsize=(12, 6))
    plt.hist(posterior_mu, bins=30, density=True, alpha=0.5, color='blue', label='Distrib a posteriori pt $\mu$')
    plt.title('Distrib a posteriori pt $\mu$')
    plt.xlabel('$\mu$')
    plt.ylabel('Densitatea de probabilitate')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
