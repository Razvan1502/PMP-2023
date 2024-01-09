import numpy as np
import matplotlib.pyplot as plt

# Ex1
# Definirea dimensiunilor grid-ului
rows = 5
cols = 5

# Generarea unui grid aleatoriu intre 0 și 1
grid = np.random.rand(rows, cols)

# dfinirea unei distributii alternative
prior = (grid <= 0.5).astype(int)

# simularea grid computing cu distributia alternativa
result = np.sum(prior)

# Afisarea rezultatului
print("Grid original:")
print(grid)
print("\nPrior distribuție:")
print(prior)
print("\nSuma cu distribuția alternativă:", result)


# Ex2


def estimate_pi(N):
    # Genereazm N puncte aleatorii într-un patrat [0, 1] x [0, 1]
    points = np.random.rand(N, 2)

    # verificm daca fiecare punct este in interiorul cercului de raza 1
    inside_circle = np.linalg.norm(points, axis=1) <= 1.0

    # Calculează estimarea lui pi
    pi_estimate = 4 * np.sum(inside_circle) / N

    # Calculam eroarea
    error = np.abs(np.pi - pi_estimate)

    return error


# Parametrii
num_simulations = 100  # Nr de simulari
N_values = [100, 1000, 10000]  # Diferite valori ale lui N

# Listele pentru stocarea rezultatelor
mean_errors = []
std_dev_errors = []

# Rulam simularile pentru diferite valori ale lui N
for N in N_values:
    errors = [estimate_pi(N) for _ in range(num_simulations)]

    # Calculeazam media si deviatia standard a erorilor
    mean_errors.append(np.mean(errors))
    std_dev_errors.append(np.std(errors))

# Vizualizam rezultatele folosind plt.errorbar()
plt.errorbar(N_values, mean_errors, yerr=std_dev_errors, fmt='o-', capsize=5)
plt.xscale('log')  # Pentru a vizualiza mai bine pe scala logaritmica
plt.xlabel('Numărul de puncte (N)')
plt.ylabel('Eroare în estimarea lui π')
plt.title('Relația dintre N și eroare în estimarea lui π')
plt.show()


# Ex3

def beta_binomial_prior(alpha, beta):
    # Functie pentru distributia a priori beta-binomiala
    pass


def metropolis(num_samples, alpha, beta):
    # Functia Metropolis cu distributie a priori beta-binomiala
    samples = np.zeros(num_samples)
    current_sample = np.random.rand()

    for i in range(num_samples):
        # Generarea unei propuneri noi folosind o distributie normala, de exemplu
        proposal = current_sample + np.random.normal(0, 0.1)

        # Calculam raportului de distributii
        acceptance_ratio = beta_binomial_prior(proposal, alpha, beta) / beta_binomial_prior(current_sample, alpha, beta)

        # Alegerea daca acceptam sau respingem propunerea
        if np.random.rand() < acceptance_ratio:
            current_sample = proposal

        samples[i] = current_sample

    return samples


# exemplu
alpha = 2  # Parametrul alpha al distribuției beta
beta = 5  # Parametrul beta al distribuției beta
num_samples = 1000

samples = metropolis(num_samples, alpha, beta)

print("Muestră finală:")
print(samples[-10:])
