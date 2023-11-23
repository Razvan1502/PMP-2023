import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

"""     AM LUAT 2.5/3
Ex. 1 (1pct.) Doi mecanici schimbă filtrele de ulei pentru autoturisme într-un service. Timpul de servire este exponenţial
cu parametrul λ1 = 4 hrs−1 în cazul primului mecanic si λ2 = 6 hrs−1 în cazul celui de al doilea. Deoarece al doilea
mecanic este mai rapid, el serveşte de 1.5 ori mai mulţi clienţi decât partenerul său. Astfel când un client ajunge la rând,
probabilitatea de a servit de primul mecanic este 40%. Fie X timpul de servire pentru un client.
Generaţi 10000 de valori pentru X, şi în felul acesta estimaţi media şi deviaţia standard a lui X. Realizaţi un grafic al
densităţii distribuţiei lui X.
Notă: Distribuţia Exp(λ) se poate apela cu stats.expon(0,1/λ) sau stats.expon(scale=1/λ).
"""

lambda1 = 4**-1 
lambda2 = 6**-1  
P1 = 0.4  # prob pt primul mecanic
P2 = 1-P1 # prob pt al doilea
n = 10000

#lista de n valori random pt servirea de catre primul mecanic ori al doilea
servire = np.random.choice([1, 2], n, p=[P1, P2])

#timpul de servire al fiecarui client
timp_servire = []

for i in range(n):
    if servire[i] == 1:
        x = stats.expon(scale=1/lambda1).rvs()
    else:
        x = stats.expon(scale=1/lambda2).rvs()
    timp_servire.append(x)

timp_servire = np.array(timp_servire)

mean_X = np.mean(timp_servire)
std_deviation_X = np.std(timp_servire)

print("Media X:", mean_X)
print("Deviatia standard X:", std_deviation_X)


#set de valori pt a reprezenta distrib lui X
x_values = np.linspace(0, timp_servire.max(), 100)
density1 = P2 * stats.expon(scale=1/lambda2).pdf(x_values) + P1 * stats.expon(scale=1/lambda1).pdf(x_values)

#densitatea distrib lui X
plt.figure(figsize=(10, 6))
plt.plot(x_values, density1, label='Densitatea lui X')
plt.xlabel('Timpul de servire (X)')
plt.ylabel('Densitate')
plt.legend()
plt.title('Densitatea distributiei lui X')
plt.grid(True)
plt.show()
