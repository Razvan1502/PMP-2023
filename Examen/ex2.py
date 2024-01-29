import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

theta_X = 0.3
theta_Y = 0.5

N = 1000

# Simulare variabile aleatoare Geometrice
X = geom.rvs(theta_X, size=N)
Y = geom.rvs(theta_Y, size=N)

# Calc  P(X > Y^2) folosind metoda Monte Carlo
count = np.sum(X > Y**2)
probability = count / N

#rezultate
plt.plot(X[Y**2 < X], Y[Y**2 < X], 'b.')  # Pct care indeplinesc x > y^2
plt.plot(X[Y**2 >= X], Y[Y**2 >= X], 'r.')  # Pct care nu îndeplinesc  x > y^2
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Aproximare P(X > Y^2): {probability:.4f}')
plt.show()


#b
k = 30

probabilities = []

for _ in range(k):
    # Simulare variabile aleatoare Geometrice
    X = geom.rvs(theta_X, size=N)
    Y = geom.rvs(theta_Y, size=N)

    count = np.sum(X > Y**2)
    probability = count / N

    #adaugam la lista de probabilitati
    probabilities.append(probability)

# Calc media  si deviatia standard
mean_probability = np.mean(probabilities)
std_deviation = np.std(probabilities)

# rezultate
plt.hist(probabilities, bins=10, edgecolor='black')
plt.xlabel('Probabilitate P(X > Y^2)')
plt.ylabel('Frecventa')
plt.title(f'Distributia probabilitatilor pt k = {k} aproximari\nMedie: {mean_probability:.4f}, Deviatie standard: {std_deviation:.4f}')
plt.show()