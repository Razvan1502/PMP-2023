import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

"""
Doreşti sa deschizi o nouă locaţie fast-food în oraş. Analizând volumul de trafic al locaţiei, aproximezi ca
numărul de clienţi care ar intra în restaurant umează o distribuţie Poisson de parametru λ = 20 clienţi/oră.
Timpul de plasare si plată a unei comenzi la o casă urmează o distribuţie normală cu media de 2 minute si
deviatie standard de 0.5 minute. O staţie de gătit pregateste o comandă intr-un timp descris de o distribuţie
exponenţială cu media de α minute.
1. Definiţi modelul probabilist care sa descrie contextul de mai sus. (0.5p - deadline: sfârşitul seminarului)
2. Determinaţi care este (cu aproximaţie) α maxim pentru a le putea servi mancarea intr-un timp mai scurt
de 15 minute tuturor clienţilor care intră într-o oră, cu o probabilitate de 95%. (1p - deadline: sfârşitul zilei de
seminar)
3. Cu α astfel calculat, determinaţi timpul mediu de aşteptare pentru a fi servit al unui client. (0.5p -
deadline: sfârşitul zilei de seminar)
"""

n_value = 100
np.random.seed(1)

alfa = 0.0
while True:
    client_wait_time = []
    good = 0
    clients_number = stats.poisson.rvs(20.0, size=n_value)
    for c_nr in clients_number:
        order_time = stats.norm.rvs(2.0, 0.5, size=c_nr) #pt un client
        preparation_time = stats.expon.rvs(loc=alfa, size=c_nr)
        wait = order_time + preparation_time

        ok = True
        for w in wait:
            if w >= 15: #daca depaseste 15 min ok=false
                ok = False
            client_wait_time.append(w) #se adauga timpul in lista

        if ok: #daca ok=true inseaman ca toti clienti au fost serviti in <15min si se incrementeaza contorul good
            good += 1

    if good/n_value <= 0.95:
        break

    mean = sum(client_wait_time) / len(client_wait_time)
    alfa += 0.001

print(f"alfa is {alfa}")
print(f"mean in {mean}")