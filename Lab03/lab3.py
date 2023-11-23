from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

"""
Un depozit de marfă este echipat cu alarmă de incendiu. Pentru o zi dată, probabilitatea să aibă loc un
cutremur in zona depozitului este de 0.05%.
Şansa ca un incendiu să se declanşeze la depozit este aproximată la 1% in mod normal dar, dacă a avut loc
un cutremur, aceasta creşte la 3%.
Alarma de incendiu are probabilitatea să se declanşeze accidental de 0.01%, dar în caz de cutremur această
declanşare accidentală urcă la 2%; în caz de incendiu alarma are 95% şansă de declanşare, iar dacă a avut loc
şi cutremur şi incendiu, alarma se declanşează cu probabilitatea de 98%.
1. Definiti modelul probabilist, folosind pgmpy, care sa descrie contextul de mai sus. (1p)
2. Ştiind că alarma de incendiu a fost declanşată, calculaţi probabilitatea să fi avut loc un cutremur. (1p)
3. Afişaţi probabilitatea ca un incendiu sa fi avut loc, fără ca alarma de incendiu să se activeze. (1p)
"""

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
