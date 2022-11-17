import numpy as np
from scipy.stats import dweibull
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

beta, m = 2, 3

c = 2.07

mean, var, skew, kurt = dweibull.stats(c, moments='mvsk')
x = np.linspace(dweibull.ppf(0.01, c),

                dweibull.ppf(0.99, c), 100)

ax.plot(x, dweibull.pdf(x, c),

        'r-', lw=5, alpha=0.6, label='dweibull pdf')

plt.show()
