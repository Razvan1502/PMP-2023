import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats

"""
Valorile de trafic (masini/min.) înregistrate de o camera in jurul unei anumite intersectii din oraş, într-o zi
normală, sunt redate în fişierul trafic.scv (în fiecare minut, de la ora 4:00 până la 24:00). Presupunem că valorile
respective sunt determinate de o distribuţie Poisson de parametru necunoscut λ > 0. Se ştie că modificări ale
mediei traficului au loc în jurul orelor 7, 16 (creştere) şi 8, 19 (descreştere).
1. Definiţi un model probabilist care sa descrie contextul de mai sus, folosind PyMC. (1pt)
2. Determinaţi capetele cele mai probabile ale celor 5 intervale de timp, cât şi cele mai probabile valori ale
parametrului λ în acele intervale. (1pt)
"""

count_data = np.loadtxt("trafic.csv", delimiter=',', dtype=int, skiprows=1)

n_count_data = len(count_data)
with pm.Model() as model:
    alpha = 1.0 / count_data[:, 1].mean()
    # Definirea a cinci variabile lambda cu distribuție exponențială ca priori
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)

    # Definirea a patru intervale de timp cu distribuție normală ca priori
    interval1 = pm.Normal("interval1", 60 * 3)
    interval2 = pm.Normal("interval2", 60 * 12)
    interval3 = pm.Normal("interval3", 60 * 15)
    interval4 = pm.Normal("interval4", 60 * 20)

    # Definirea a patru puncte de schimbare discrete în timp
    tau1 = pm.DiscreteUniform("tau1", lower=1, upper=interval1) #4-7
    tau2 = pm.DiscreteUniform("tau2", lower=tau1, upper=interval2) #7-16
    tau3 = pm.DiscreteUniform("tau3", lower=tau2, upper=interval3) #16-19
    tau4 = pm.DiscreteUniform("tau4", lower=tau3, upper=interval4) #19-24

    # Definirea unor variabile de index pentru a crea intervale de timp
    idx = np.arange(n_count_data)
    lmbd1 = pm.math.switch(tau1 > idx, lambda_1, lambda_2)
    lmbd2 = pm.math.switch(tau2 > idx, lmbd1, lambda_3)
    lmbd3 = pm.math.switch(tau3 > idx, lmbd2, lambda_4)
    lmbd4 = pm.math.switch(tau4 > idx, lmbd3, lambda_5)
    # Definirea variabilei observate (likelihood) ca distribuție Poisson
    observation = pm.Poisson("obs", lmbd4, observed=count_data[:, 1])
    trace = pm.sample(10, cores=1)
    az.plot_posterior(trace)
    plt.show()


"""
Vom folosi modelul descris în problema din laboratorul 4
(pct.1).
1. Pentru α = 3, generaţi mai întâi un eşantion de 100 de timpi de aşteptare medii (un timp de aşteptare
mediu este media timpilor de aşteptare ai tuturor clienţilor ce intră în decurs de o oră) (0.5pt).
2. Creaţi un model echivalent în PyMC pentru a putea infera cu datele de mai sus asupra lui α. Cu ajutorul
unui grafic KDE sau al unui sumar al datelor eşantionate, explicaţi dacă rezultatul obţinut corespunde valorii
aşteptate (α = 3). (0.5pt).
"""

client_nr = stats.poisson.rvs(20, size=1)

command_nr = stats.norm.rvs(loc=2, scale=0.5, size=20)

# Funcția calc determină valoarea optimă a lui α astfel încât 95% dintre clienți să aibă un timp de așteptare mai mic de 15 minute
def calc(guess):
    rand_gen = stats.expon.rvs(loc=guess, size=20)
    count = 0
    for element in rand_gen:
        if element < 15:
            count = count + 1
    if count / 20 * 100 > 95:
        return calc(guess + 1)
    else:
        return guess


alpha = calc(1)
print(alpha)

# Generarea unui eșantion de 100 de timpi de așteptare medii folosind α = 3
meanlist = []
for i in range(100):
    cook_nr = stats.expon.rvs(loc=3, size=20).mean()
    meanlist.append(copy.deepcopy(cook_nr))

with pm.Model() as model:
    alpha = 3
    nr_clienti = pm.Poisson("nr_clienti", mu=20)
    timpPlasarePLata = pm.Normal("timpPlasarePlata", mu=2, sigma=0.5)
    timpPregatire = pm.Exponential("timpPregatire", lam=1 / alpha)

    observation = pm.Poisson("obs", mu=timpPregatire, observed=meanlist)


    with model:
        trace = pm.sample(1000, cores=1)
        az.plot_posterior(trace)
        plt.show()