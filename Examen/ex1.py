import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

titanic_data = pd.read_csv("Titanic.csv")

print(titanic_data.head())

# Gestionarea valorilor lipsa
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)


data_subset = titanic_data[["Survived", "Pclass", "Age"]]

# setul de date preprocesat
print(data_subset.head())

#b Definirea modelului

if __name__ == '__main__':
    with pm.Model() as logistic_model:
        # Coeficienți
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=data_subset.shape[1] - 1)

        # Model liniar
        X_shared = pm.Data('X_shared', data_subset[["Pclass", "Age"]].values)
        mu = pm.Deterministic('mu', alpha + pm.math.dot(X_shared, beta))

        # Regresie logistica
        theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
        #granita de decizie
        bd = pm.Deterministic('bd', -alpha / beta[1] - beta[0] / beta[1] * data_subset.iloc[1:, 0])


        # Predictie Bernoulli
        y_pred = pm.Bernoulli("y_pred", p=theta, observed=data_subset["Survived"].values)

        idata = pm.sample(2000, return_inferencedata=True)


        az.plot_posterior(idata, var_names=['alpha','beta'])
        plt.show()

#c Variabila beta1 - Age infuenteaza mai mult pentru ca are o mmedie mai mare

#d
def compute_probability(Pclass,age):
  prob = []
  alpha0=idata.posterior['alpha'][1]
  betas = idata.posterior['beta'][1] #beta1 si beta2

  for i in range(len(alpha0)):
      prob.append(1 / (1 + np.exp(-(alpha0[i] + Pclass * betas[i][0] + age * betas[i][1])))) #probabilitatea de supravietuire
  prob = np.array(prob)
  intervals = pm.stats.hdi(prob, hdi_prob=0.90) #intervalul HDI

  hdi_prob = []
  for i in range(0,len(prob)):
      if prob[i] >= intervals[0] and prob[i] <= intervals[1]:
          hdi_prob.append(prob[i])

  hdi_prob = np.array(hdi_prob)
  hdi_prob = np.sort(hdi_prob)
  plt.figure()
  plt.plot(hdi_prob)
  plt.title(f'Pclas = {Pclass} Age = {age}')
  plt.show()

  compute_probability(2, 30)