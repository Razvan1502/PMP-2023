import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

import random


def arunca_moneda(jucator, nr_aruncari):
    if jucator == 0:
        # moneda nemasluita  j0
        rezultate = np.random.randint(2, size=nr_aruncari)
    else:
        # Moneda masluita   j1
        rezultate = [random.choices([0, 1], weights=[1 / 3, 2 / 3])[0] for _ in range(nr_aruncari)]

    return rezultate


def simulare_joc():
    #cine incepe
    jucator_curent = np.random.randint(2)

    rezultate_j0 = arunca_moneda(0, 1)

    rezultate_j1 = arunca_moneda(1, len(rezultate_j0) + 1)


    castigator = 0 if sum(rezultate_j0) >= sum(rezultate_j1) else 1

    return castigator


numar_simulari = 10000
castiguri_j0 = sum(simulare_joc() == 0 for _ in range(numar_simulari))
castiguri_j1 = sum(simulare_joc() == 1 for _ in range(numar_simulari))

print(f"Sansele  pt j0: {castiguri_j0 / numar_simulari}")
print(f"Sansele pt j1: {castiguri_j1 / numar_simulari}")



#2

model = BayesianNetwork([('jucator0', 'castigator'), ('jucator1', 'castigator')])

cpd_jucator0 = TabularCPD('jucator0', 2, np.array([[0.5], [0.5]]))
cpd_jucator1 = TabularCPD('jucator1', 2, np.array([[1/3], [2/3]]))


#var castigator in funct de j0 si j1
cpd_castigator = TabularCPD('castigator', 2, np.array([[0.0, 1.0], [1.0, 0.0]]),
                            evidence=['jucator0', 'jucator1'], evidence_card=[2, 2])

model.add_cpds(cpd_jucator0, cpd_jucator1, cpd_castigator)

model.check_model()

inference = VariableElimination(model)
result = inference.query(variables=['castigator'], evidence={'jucator0': 0, 'jucator1': 1})
print(result)

