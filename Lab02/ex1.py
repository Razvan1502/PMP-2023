import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1) 

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
