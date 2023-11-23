import pymc as pm
import matplotlib.pyplot as plt

""" Am luat 2/2
Un magazin este vizitat de n clienţi într-o anumită zi. Numărul Y de clienţi care cumpără un anumit produs
e distribuit Binomial(n, θ), unde θ este probabilitatea ca un client să cumpere acel produs. Să presupunem că
îl cunoaştem pe θ şi că distribuţia a priori pentru n este Poisson(10).
1. (1pt) Folosiţi PyMC pentru a calcula distribuţia a posteriori pentru n pentru toate combinaţiile de
Y ∈ {0, 5, 10} şi θ ∈ {0.2, 0.5}. Folosiţi az.plot_posterior pentru a vizualiza toate rezultatele
(ideal, într-o singură fereastră).
2. (1pt) Explicaţi efectul lui Y şi θ asupra distribuţiei a posteriori.
"""


Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

fig, axs = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 10))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:
            # Distrib a priori pt n
            n = pm.Poisson('n', mu=10)

            # Distrib binomiala pt Y nr de clienti care cumpara
            y = pm.Binomial('Y', n=n, p=theta, observed=Y)

            trace = pm.sample(2000, tune=1000, cores=1)
           # afiseaza distrib a posteriori
        pm.plot_posterior(trace, ax=axs[i, j])
        axs[i, j].set_title(f'Y = {Y}, θ = {theta}')

plt.tight_layout()
plt.show()
