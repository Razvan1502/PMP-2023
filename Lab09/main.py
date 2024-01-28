import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

"""Am luat 1/2 pct , nu facusem c si d 

Ce factori determină admiterea la facultate în Statele Unite? În fişierul Admission.csv au fost strânse datele
a 400 de cazuri de admitere la o facultate. “Admission” este un răspuns binar, cu 1 sau 0 indicând “admis”,
respectiv “respins”. Sunt de asemenea disponibile scorul la testul GRE şi rezultatul mediu din liceu, GPA
(undergraduate grade point average). Fie pi probabilitatea ca studentul cu nr. i să fie admis.
Considerăm modelul logistic pi = logistic(β0 + β1xi1 + β2xi2),

unde xi1 şi xi2 sunt scorul GRE, respectiv GPA pentru studentul i.
1. Folosind distribuţii a priori slab informative asupra parametrilor β0, β1 şi β2, folosiţi PyMC pentru a
simula un eşantion suficient de mare (construi modelul) din distribuţia a posteriori. (0.5p)
2. Care este, în medie, graniţa de decizie pentru acest model? Reprezentaţi de asemenea grafic o zonă în
jurul acestei grafic care să reprezinte un interval 94% HDI. (0.5p)
3. Să presupunem că un student are un scor GRE de 550 şi un GPA de 3.5. Construiţi un interval de 90%
HDI pentru probabilitatea ca acest student să fie admis. (0.5p)
4. Dar dacă studentul are un scor GRE de 500 şi un GPA de 3.2? (refaceţi exerciţiul anterior cu aceste date)
Cum justificaţi diferenţa? (0.5p)
"""



raw_data = pd.read_csv('./Admission.csv')
data = raw_data[["Admission", "GRE", "GPA"]]
y_1 = data['Admission'].values

x_n = ['GPA', 'GRE']
x_1 = np.array(data[x_n].values,dtype=np.float64)

#a
with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sigma=10) # beta_0
    beta = pm.Normal('beta', mu=0, sigma=2, shape=len(x_n)) # beta_1 si beta_2
    mu = alpha + pm.math.dot(x_1, beta) # probabilitatea de admitere
    teta = pm.Deterministic('teta', 1 / (1 + pm.math.exp(-mu))) # probabilitatea de admitere
    bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:,0]) # granița de decizie
    yl = pm.Bernoulli('yl', p=teta, observed=y_1) # yl = 1 daca studentul este admis, 0 altfel
    idata_1 = pm.sample(50, cores=1,return_inferencedata=True) # 50 de esantioane din distributia a posteriori

az.plot_posterior(idata_1, var_names=['alpha','beta'])
plt.show()

#b
idx = np.argsort(x_1[:,0]) # sortam dupa GPA
bd = idata_1.posterior['bd'].mean(("chain", "draw"))[idx] # granița de decizie
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_1]) # afisam punctele
plt.plot(x_1[:,0][idx], bd, color='k'); # afisam granița de decizie
az.plot_hdi(x_1[:,0], idata_1.posterior['bd'], color='k',hdi_prob=0.94) # afisam intervalul HDI
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
plt.show()

# pct c si d
def compute_probability(gre,gpa):
  prob = []
  alpha0=idata_1.posterior['alpha'][1]
  betas = idata_1.posterior['beta'][1]

  for i in range(len(alpha0)):
      prob.append(1 / (1 + np.exp(-(alpha0[i] + gpa * betas[i][0] + gre * betas[i][1]))))
  prob = np.array(prob)
  intervals = pm.stats.hdi(prob, hdi_prob=0.90)

  hdi_prob = []
  for i in range(0,len(prob)):
      if prob[i] >= intervals[0] and prob[i] <= intervals[1]:
          hdi_prob.append(prob[i])

  hdi_prob = np.array(hdi_prob)
  hdi_prob = np.sort(hdi_prob)
  plt.figure()
  plt.plot(hdi_prob)
  plt.title(f'GRE = {gre} GPA = {gpa}')
  plt.show()


compute_probability(550,3.5)
compute_probability(500,3.2)

# in ambele cazuri studentul nu va fi admis, avand o probabilitatea prea mica de a fi admis
# primul student are o probabilitate mai mare de a fi admis dar diferenta nu este foarte semnificativa,
# acest lucru putand fi datorat inconsistentelor din setul de date