import pymc as pm
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

fig, axs = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 10))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:
            # Distrib a priori pt n
            n = pm.Poisson('n', mu=10)

            # Distrib binomiala pt Y
            y = pm.Binomial('Y', n=n, p=theta, observed=Y)

            trace = pm.sample(2000, tune=1000, cores=1)
           # afiseaza distrib a posteriori
        pm.plot_posterior(trace, ax=axs[i, j])
        axs[i, j].set_title(f'Y = {Y}, θ = {theta}')

plt.tight_layout()
plt.show()
