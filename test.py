import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(10**-6, 1, 100)
for gamma in range(10):
    f = np.log(x) + gamma * x ** 2
    plt.plot(x, f, label=r'$\gamma = %d$' % gamma)
plt.legend()
plt.show()