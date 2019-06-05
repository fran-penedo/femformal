from __future__ import division, absolute_import, print_function

import numpy as np

N = 20
L = 100000
rho_steel = 8e-6
rho_brass = 8.5e-6
E_steel = 200e6
E_brass = 100e6
rho = lambda x: rho_steel if x < 30000 or x > 60000 else rho_brass
E = lambda x: E_steel if x < 30000 or x > 60000 else E_brass
xpart = np.linspace(0, L, N + 1)
# f_nodal[-1] = 2e6

f_nodal = np.zeros(N + 1)
u0 = lambda x: 0.0
du0 = lambda x: 0.0

g = [0.0, None]

dt = .0025
T = 0.5

N_t = 200
dt_t = 0.0025
f_nodal_t = np.zeros(N_t + 1)
xpart_t = np.linspace(0, L, N_t + 1)
