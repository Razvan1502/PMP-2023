import pymc3 as pm
import numpy as np


lmbda = 20  # clienti/ora
mean_order_time = 2 
std_dev_order_time = 0.5

time_limit = 15  
probability_limit = 0.95

# distrib. Poisson pt nr clienti
with pm.Model() as model:
    num_clients = pm.Poisson("num_clients", mu=lmbda)

# distrib. normala pt timpul de plasare si plata comenzii
with model:
    order_time = pm.Normal("order_time", mu=mean_order_time, sigma=std_dev_order_time)

# distrib. exponentiala pt timpul de pregatire
with model:
    alpha = pm.Uniform("alpha", lower=0, upper=10)  
    cook_time = pm.Exponential("cook_time", lam=alpha)

# timp total pt un client
with model:
    total_time = order_time + cook_time

# ex2
with model:
    # probabilitatea ca timpul  pt un client este < de time_limit
    prob_total_time = 1 - pm.math.exp(-total_time)
    
    # probabilitatea >= cu probability_limit
    observed = pm.Potential("observed", -pm.math.log(1 - probability_limit - (1 - prob_total_time)))

# sampling Bayesian
with model:
    trace = pm.sample(10000, tune=1000, chains=2)

pm.summary(trace)

#  alpha maxim
alpha_max_estimated = trace["alpha"].max()

print(f"α maxim este: {alpha_max_estimated:.4f}")

# timpul mediu pt pregtirea comenzii
mean_cook_time = 1 / alpha_max_estimated

# timpul mediu de asteptare pt un client
mean_waiting_time = mean_order_time + mean_cook_time

print(f"timpul mediu de asteptare pt a fi servit un client este {mean_waiting_time:.2f} minute")