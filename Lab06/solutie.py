import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
from scipy import stats

"""
Un magazin este vizitat de n clienţi într-o anumită zi. Numărul Y de clienţi care cumpără un anumit produs
e distribuit Binomial(n, θ), unde θ este probabilitatea ca un client să cumpere acel produs. Să presupunem că
îl cunoaştem pe θ şi că distribuţia a priori pentru n este Poisson(10).
1. (1pt) Folosiţi PyMC pentru a calcula distribuţia a posteriori pentru n pentru toate combinaţiile de
Y ∈ {0, 5, 10} şi θ ∈ {0.2, 0.5}. Folosiţi az.plot_posterior pentru a vizualiza toate rezultatele
(ideal, într-o singură fereastră).
2. (1pt) Explicaţi efectul lui Y şi θ asupra distribuţiei a posteriori.
"""

'''
Din grafice se observa urmatoarele:

Caz 1: Numarul de clienti care cumpara un produs ramane neschimbat
    Numarul total de clienti scade daca probabilitatea de a cumpara un produs creste
Caz 2: Probabilitatea de a cumpara un produs ramane neschimbata
    Numarul total de clienti creste daca numarul de clienti care cumpara un produs creste

De asemenea numarul total de clienti este afectat mai puternic de cresterea numarului de clienti care
cumpara un produs si mai putin de probabilitatea de a cumpara un produs
'''

y=[0,5,10]
teta=[0.2, 0.5]

posteriors=[]
for i in range(0, len(y)):
    for j in range(0, len(teta)):
        with pm.Model() as model:
            n= pm.Poisson("n", mu=10)
            pm.Binomial("buyers"+str(i)+" "+str(j),n=n,p=teta[j],observed=y[i])
            idata_t = pm.sample(100, return_inferencedata=True,cores=1)
            posteriors.append((y[i], teta[j], idata_t))


fig, axes = plt.subplots(nrows=len(y), ncols=len(teta), figsize=(12, 8))

for i, (Y, theta, trace) in enumerate(posteriors):
    ax = axes[i // len(teta), i % len(teta)]
    # print(i // len(teta), i % len(teta))
    az.plot_posterior(trace,var_names=['n'], ax=ax)
    ax.set_title(f'Y={Y}, θ={theta}')

plt.tight_layout()
plt.show()
