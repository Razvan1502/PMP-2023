import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(1)  

alpha1, lambda1 = 4, 3
alpha2, lambda2 = 4, 2
alpha3, lambda3 = 5, 2
alpha4, lambda4 = 5, 3

lambda_latency = 4

p_server1 = 0.25
p_server2 = 0.25
p_server3 = 0.30
p_server4 = 1 - (p_server1 + p_server2 + p_server3)

n_simulations = 100000 


service_times = []

for _ in range(n_simulations):
    alegerea = np.random.choice([1, 2, 3, 4], p=[p_server1, p_server2, p_server3, p_server4])

    if alegerea == 1:
        timp_procesare = stats.gamma(alpha1, scale=1/lambda1).rvs()
    elif alegerea == 2:
        timp_procesare = stats.gamma(alpha2, scale=1/lambda2).rvs()
    elif alegerea == 3:
        timp_procesare = stats.gamma(alpha3, scale=1/lambda3).rvs()
    else:
        timp_procesare = stats.gamma(alpha4, scale=1/lambda4).rvs()





