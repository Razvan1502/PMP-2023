import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
"""
Se consideră un experiment aleator prin aruncarea de 10 ori a două monezi, una nemăsluită, cealaltă cu
probabilitatea de 0.3 de a obţine stemă. Să se genereze 100 de rezultate independente ale acestui experiment şi astfel
să se determine grafic distribuţiile variabilelor aleatoare care numără rezultatele posibile în cele 10 aruncări (câte una
pentru fiecare rezultat posibil: ss, sb, bs, bb).
"""

# Solutia cu triviala prin generarea de monezi si iterarea + calcularea distributiilor manual
bb = []
bs = []
sb = []
ss = []
for i in range(100):
    stema_moneda1 = stats.binom.rvs(1,0.5, size=10) #lista de 10 var aleatoare binomiale n=1 (nr de incercari) și p=0.5 (probab pt stema)
    stema_moneda2 = stats.binom.rvs(1,0.3, size=10)
    bb_c = 0
    bs_c = 0
    sb_c = 0
    ss_c = 0
    for j in range(10):
        if stema_moneda1[j] == 0 and stema_moneda2[j] == 0:
            bb_c += 1
        elif stema_moneda1[j] == 0 and stema_moneda2[j] == 1:
            bs_c += 1
        elif stema_moneda1[j] == 1 and stema_moneda2[j] == 0:
            sb_c += 1
        elif stema_moneda1[j] == 1 and stema_moneda2[j] == 1:
            ss_c += 1
    bb.append(bb_c)
    bs.append(bs_c)
    sb.append(sb_c)
    ss.append(ss_c)

az.plot_posterior({'bb':bb,'bs':bs,'sb':sb,'ss':ss})
plt.show()

# Solutii mai compacte care generau direct distributiile combinatiilor sunt si ele corecte
# Ex:
X = stats.multinomial.rvs(10,[0.5*0.7,0.5*0.3,0.5*0.7,0.5*0.3], size=100) # Generat probabilitatile pentru combinatiile de monezi

az.plot_posterior({'bb':X[:,0],'bs':X[:,1],'sb':X[:,2],'ss':X[:,3]})
plt.show()