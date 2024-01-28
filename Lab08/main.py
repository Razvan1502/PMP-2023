
import math

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

"""  Am luat 1.5/3 puncte . Am facut doar primele 3 cerinte. 
Ce factori determinau preţul unui PC în vremea apariţiei acestora? Fişierul Prices.csv descrie un eşantion
de 500 de vânzări de PC, colectat din 1993 până în 1995 în Statele Unite. Pe lângă preţul de vânzare, au fost
colectate informaţii despre frecvenţa procesorului în MHz, dimensiunea hard diskului în MB, dimensiunea
Ram în MB şi dacă producătorul este premium (de exemplu, IBM, COMPAQ). Să presupunem că se ia în
considerare modelul de regresie y ∼ N (μ, σ), unde μ = α + β1x1 + β2x2,

y este preţul de vânzare, x1 este frecvenţa procesorului şi x2 este logaritmul (natural) al mărimii hard diskului.
1. Folosind distribuţii a priori slab informative asupra parametrilor α, β1, β2 şi σ, folosiţi PyMC pentru a
simula un eşantion suficient de mare din distribuţia a posteriori. (1p)
2. Obţineţi estimări de 95% HDI ale parametrilor β1 şi β2. (0.5p)
3. Pe baza rezultatelor obţinute, sunt frecvenţa procesorului şi mărimea hard diskului predictori utili ai
preţului de vânzare? (0.5p)
4. Să presupunem acum că un consumator este interesat de un computer cu o frecvenţă de 33 MHz şi un
hard disk de 540 MB. Simulaţi 5000 de extrageri din preţul de vânzare aşteptat (μ) şi construiţi un interval de
90% HDI pentru acest preţ. (0.5p)
5. În schimb, să presupunem că acest consumator doreşte să prezică preţul de vânzare al unui computer cu
această frecvenţă şi mărime a hard disk-ului. Simulaţi 5000 de extrageri din distribuţia predictivă posterioară
şi utilizaţi aceste extrageri simulate pentru a găsi un interval de predicţie de 90% HDI. (0.5p)
Bonus. Afectează în vreun fel preţul faptul ca producătorul este premium? Justificaţi. (1p) 
"""

raw_data = pd.read_csv('./Prices.csv')
data = raw_data[['Price', 'Speed', 'HardDrive', 'Ram', 'Premium']]
price=data['Price'].values
x1 = data['Speed'].values # frecventa procesorului
x2 = data['HardDrive'].values # dimensiunea hard diskului

price=np.array(price,dtype=np.float64)
x1 = np.array(x1, dtype=np.float64)
x2 = np.array(x2, dtype=np.float64)
x2 = np.log(data['HardDrive'])



with pm.Model() as model_g:
    alpha = pm.Normal('alpha', mu=0, sigma=10)  # alpha
    beta1 = pm.Normal('beta1', mu=0, sigma=1)
    beta2 = pm.Normal('beta2', mu=0, sigma=1)   #beta_1 si beta_2
    eps = pm.HalfCauchy('eps', 5) # sigma
    mu = pm.Deterministic('mu', alpha + beta1 * x1 + beta2 * x2 ) # mu
    price_pred = pm.Normal('price_pred', mu=mu, sigma=eps, observed=price) #
    idata_g = pm.sample(50, tune=50,cores=1, return_inferencedata=True)

az.plot_posterior(idata_g, var_names='beta1', hdi_prob=0.95) # intervalul HDI pentru beta1
plt.show()


az.plot_posterior(idata_g, var_names='beta2', hdi_prob=0.95) # intervalul HDI pentru beta2
plt.show()

mean_beta1 = idata_g.posterior['beta1'].mean().item()
mean_beta2 = idata_g.posterior['beta2'].mean().item()

print(f"beta1: {mean_beta1}")
print(f"beta2: {mean_beta2}")


# Media lui beta1 este mai mare, astfel Speed va avea o influenta mai mare fata de price in comparatie cu HardDrive

