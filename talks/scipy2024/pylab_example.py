"""git commit diff example."""

import matplotlib.pyplot as plt
import numpy as np

x = np.random.random(20)
y = np.random.random(20)

# Lines on top of scatter
plt.figure()
plt.subplot(211)
plt.plot(x, y, "r", lw=3)
plt.scatter(x, y, s=120)
