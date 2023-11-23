import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az

"""Preziceţi consumul unei maşini (în mile/galon: mpg) pe baza cailor putere (CP). Fişierul auto-mpg.csv
conţine un set de date cu observaţii, din care ne interesează doar cele corespunzătoare cailor putere şi valorile
mpg corespunzătoare.
a. (0.5p.) Încărcaţi setul de date într-un Pandas DataFrame şi trasaţi un grafic (după eventuala curăţare
a datelor) pentru a vizualiza relaţia de dependenţă dintre cele două variabile: CP şi mpg.
b. (1p.) Definiţi modelul în PyMC folosind CP ca variabilă independentă şi mpg ca variabilă dependentă.
c. (1p.) Determinaţi care este dreapta de regresie care se potriveşte cel mai bine datelor.
d. (0.5p.) Adăugaţi graficului de la punctul a regiunea 95%HDI pentru distribuţia predictivă a posteriori.
Ce concluzie puteţi trage asupra modelului? """

df = pd.read_csv('auto-mpg.csv')

data=df[['mpg', 'horsepower']]

data=data.drop(data[data.horsepower == '?'].index) #elimina val care contin ?
print(data)


y = data['mpg'].values
x = data['horsepower'].values

y=np.array(y,dtype=np.float64)
x=np.array(x,dtype=np.float64)

#a
plt.scatter(x, y)
plt.title('Relatia dintre horsepower si mpg')
plt.xlabel('mpg')
plt.ylabel('horsepower', rotation=0)
plt.show()

#b
with pm.Model() as model_g:
    alpha = pm.Normal('alpha', mu=y.mean(), sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    mu = pm.Deterministic('mu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y)
    idata_g = pm.sample(500, return_inferencedata=True,cores=1)



#c
plt.plot(x, y, 'C0.')
posterior_g = idata_g.posterior.stack(samples={"chain", "draw"}) #distrib posteriori
alpha_m = posterior_g['alpha'].mean().item() #media alfa
beta_m = posterior_g['beta'].mean().item() #media beta
draws = range(0, posterior_g.samples.size, 10)

plt.plot(x, posterior_g['alpha'][draws].values
         + posterior_g['beta'][draws].values * x[:,None], c='gray', alpha=0.5) #diversele drepte de regresie obținute

plt.plot(x, alpha_m + beta_m * x, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x') #traseaza dr de regresie medie
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

plt.show()

#d
ppc = pm.sample_posterior_predictive(idata_g, model=model_g)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k',
               label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')

az.plot_hdi(x, ppc.posterior_predictive['y_pred'], hdi_prob=0.95, color='gray')
az.plot_hdi(x, ppc.posterior_predictive['y_pred'], color='gray')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.show()

#Putem spune ca CP si mpg sunt invers proportionale