import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

"""
Se consideră un experiment aleator prin aruncarea de 10 ori a două monezi, una nemăsluită, cealaltă cu
probabilitatea de 0.3 de a obţine stemă. Să se genereze 100 de rezultate independente ale acestui experiment şi astfel
să se determine grafic distribuţiile variabilelor aleatoare care numără rezultatele posibile în cele 10 aruncări (câte una
pentru fiecare rezultat posibil: ss, sb, bs, bb).
"""

prob_stema_moneda1 = 0.5  # moneda nemasluita
prob_stema_moneda2 = 0.3  

nr_exper = 100  # nr de experimente independente
n_coin_tosses = 10  # nr de aruncari


ss_counts = []   # nr perechi stema, stema
sb_counts = []  # stema, ban
bs_counts = []  # ban, stema
bb_counts = []  # ban, ban

#numaram rezultatele pentru fiecare experiment
for _ in range(nr_exper):
    rezultat = np.random.choice([0, 1], size=(n_coin_tosses, 2), p=[1 - prob_stema_moneda1, prob_stema_moneda1])
    ss_count = np.sum(np.all(rezultat == [1, 1], axis=1))
    sb_count = np.sum(np.all(rezultat == [1, 0], axis=1))
    bs_count = np.sum(np.all(rezultat == [0, 1], axis=1))
    bb_count = np.sum(np.all(rezultat == [0, 0], axis=1))
    
    ss_counts.append(ss_count)
    sb_counts.append(sb_count)
    bs_counts.append(bs_count)
    bb_counts.append(bb_count)

#frecventele relative ale rezultatelor
ss_freq = np.array(ss_counts) / n_coin_tosses
sb_freq = np.array(sb_counts) / n_coin_tosses
bs_freq = np.array(bs_counts) / n_coin_tosses
bb_freq = np.array(bb_counts) / n_coin_tosses

#graficul distributiilor
plt.figure(figsize=(10, 6))
plt.hist(ss_freq, bins=11, alpha=0.5, label='ss')
plt.hist(sb_freq, bins=11, alpha=0.5, label='sb')
plt.hist(bs_freq, bins=11, alpha=0.5, label='bs')
plt.hist(bb_freq, bins=11, alpha=0.5, label='bb')
plt.xlabel('Frecventa relativa')
plt.ylabel('Nr de experimente')
plt.legend()
plt.title('Distributiile variabilelor aleatoare')
plt.grid(True)
plt.show()
