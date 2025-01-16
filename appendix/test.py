"""version from 15.12.23
author: Oliver Irtenkauf

features: Coporate Design Colors of University Konstanz
and inverse colors for more contrast

"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 1001)
y = 3 * x
plt.plot(x, y)
plt.plot(x, x**2)
