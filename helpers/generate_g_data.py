import numpy as np
import legwork as lw

n_range = np.arange(1, 10000 + 1).astype(int)
e_range = np.linspace(0, 1, 1000)

N, E = np.meshgrid(n_range, e_range)

g_vals = lw.utils.peters_g(N, E)

np.save("../src/legwork/peters_g.npy", g_vals)
