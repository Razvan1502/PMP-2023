import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

""" Am luat 2/3 pct
Pe modelul polinomial din curs, în codul care generează datele (din fişierul date.csv), schimbaţi order=2
cu o altă valoare, de exemplu order=5.
a. Faceţi apoi inferenţa cu model_p şi reprezentaţi grafic această curbă. (0.5p)
b. Repetaţi, dar folosind o distribuţie pentru beta cu sd=100 în loc de sd=10. În ce fel sunt curbele
diferite? Încercaţi acest lucru şi cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1]). (0.5p)
2. Repetaţi exerciţiul precedent, dar creşteţi numărul de date la 500 de puncte. (1p)
3. Faceţi inferenţa cu un model cubic (order=3), calculaţi WAIC şi LOO, reprezentaţi grafic rezultatele şi
comparaţi-le cu modelele liniare şi pătratice. (1p)
"""

az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt('./data/dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

if __name__ == '__main__':
    try:
        with pm.Model() as model_l:
            α = pm.Normal('α', mu=0, sigma=1)
            β = pm.Normal('β', mu=0, sigma=10)
            ε = pm.HalfNormal('ε', 5)
            μ = α + β * x_1s[0]
            y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
            idata_l = pm.sample(2000, return_inferencedata=True)

        with pm.Model() as model_p_sd_10:
            α = pm.Normal('α', mu=0, sigma=1)
            β = pm.Normal('β', mu=0, sigma=10, shape=order)
            ε = pm.HalfNormal('ε', 5)
            μ = α + pm.math.dot(β, x_1s)
            y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
            idata_p_sd_10 = pm.sample(2000, return_inferencedata=True)

        # model_l
        x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100) #
        α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
        β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
        y_l_post = α_l_post + β_l_post * x_new
        plt.plot(x_new, y_l_post, 'C1', label='linear model')

        # model_p_sd_10
        α_p_post = idata_p_sd_10.posterior['α'].mean(("chain", "draw")).values
        β_p_post = idata_p_sd_10.posterior['β'].mean(("chain", "draw")).values
        idx = np.argsort(x_1s[0])
        y_p_post = α_p_post + np.dot(β_p_post, x_1s)
        plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
        plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
        plt.legend()

        with pm.Model() as model_sd_100:
            α = pm.Normal('α', mu=0, sigma=1)
            β = pm.Normal('β', mu=0, sigma=100, shape=order)
            ϵ = pm.HalfNormal('ϵ', 5)
            µ = α + pm.math.dot(β, x_1s)
            y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
            idata_p_sd_100 = pm.sample(10, tune=10, return_inferencedata=True)

        # model_p_sd_100
        α_p_post = idata_p_sd_100.posterior['α'].mean(("chain", "draw")).values
        β_p_post = idata_p_sd_100.posterior['β'].mean(("chain", "draw")).values
        idx = np.argsort(x_1s[0])
        y_p_post = α_p_post + np.dot(β_p_post, x_1s)
        plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {order}')
        plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
        plt.legend()

        arr = np.array([10, 0.1, 0.1, 0.1, 0.1])
        with pm.Model() as model_sd_np:
            α = pm.Normal('α', mu=0, sigma=1)
            β = pm.Normal('β', mu=0, sigma=arr, shape=order)
            ϵ = pm.HalfNormal('ϵ', 5)
            µ = α + pm.math.dot(β, x_1s)
            y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
            idata_sd_arr = pm.sample(10, tune=10, return_inferencedata=True)

        # array
        α_p_post = idata_sd_arr.posterior['α'].mean(("chain", "draw")).values
        β_p_post = idata_sd_arr.posterior['β'].mean(("chain", "draw")).values
        idx = np.argsort(x_1s[0])
        y_p_post = α_p_post + np.dot(β_p_post, x_1s)
        plt.plot(x_1s[0][idx], y_p_post[idx], 'C4', label=f'array')
        plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
        plt.legend()

    except Exception as e:
        print(f"An error occurred: {e}")
