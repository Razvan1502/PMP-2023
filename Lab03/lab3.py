from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


model = BayesianNetwork([('cutremur', 'incendiu'), ('cutremur', 'alarma'), ('incendiu', 'alarma')])

# tabelele CPD
cpd_cutremur = TabularCPD(variable='cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='incendiu', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], evidence=['cutremur'], evidence_card=[2])
cpd_alarmă = TabularCPD(variable='alarma', variable_card=2, values=[[0.9999, 0.05, 0.98 , 0.02], [0.0001, 0.95, 0.02, 0.98]], evidence=['incendiu', 'cutremur'], evidence_card=[2, 2])

# adaugam CPD la retea
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă)

# verif daca reteaua bayesiana este valida
model.check_model()

# obiect pt inferenta
inference = VariableElimination(model)

# probabil. ca a avut loc un cutremur daca alarma a fost declansata
result = inference.query(variables=['cutremur'], evidence={'alarma': 1})

print(result)
#daca alarma  nu a fost declansata
result = inference.query(variables=['incendiu'] , evidence={'alarma': 0})

print(result)
