import pymc3 as pm
import pandas as pd

"Am luat 0.5/2"

trafic = pd.read_csv(r'D:\PMP\PMP-2023-2024\Lab05\trafic.csv')
valori = trafic['nr. masini'].values

with pm.Model() as model:
    lambda_ = pm.Exponential('lambda_', 1) 

    #actualizare lambda in fct de ore
    for minute in range(len(trafic)):
        if (minute >= 420 and minute < 481) or (minute >= 960 and minute < 1021):  # Ora 7, 16
            lambda_ = lambda_ + pm.Exponential(f'lambda_creste_{minute}', 1)
        elif (minute >= 480 and minute < 561) or (minute >= 1140 and minute < 1201):  # Ora 8, 19
            lambda_ = lambda_ - pm.Exponential(f'lambda_descreste_{minute}', 1)

    observations = pm.Poisson('observations', lambda_, observed=valori)

    trace = pm.sample(1000) 

pm.summary(trace)

