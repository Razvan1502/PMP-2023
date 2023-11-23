import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(1)

""" incomplet
Patru servere web oferă acelaşi serviciu (web) clienţilor . Timpul necesar procesării unei cereri (request)
HTTP este distribuit Γ(4, 3) pe primul server, Γ(4, 2) pe cel de-al doilea, Γ(5, 2) pe cel de-al treilea, şi Γ(5, 3) pe cel de-al
patrulea (în milisecunde). La această durată se adaugă latenţa dintre client şi serverele pe Internet, care are o distribuţie
exponenţială cu λ = 4 (în miliseconde−1). Se ştie că un client este direcţionat către primul server cu probabilitatea 0.25,
către al doilea cu probabilitatea 0.25, iar către al treilea server cu probabilitatea 0.30. Estimaţi probabilitatea ca timpul
necesar servirii unui client, notat cu X, (de la lansarea cererii până la primirea răspunsului) să fie mai mare decât 3
milisecunde. Realizaţi un grafic al densităţii distribuţiei lui X.
Notă: Distribuţia Γ(α, λ) se poate apela cu stats.gamma(α,0,1/λ) sau stats.gamma(α,scale=1/λ).
"""

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





