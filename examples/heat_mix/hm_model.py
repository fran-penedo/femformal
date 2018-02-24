import numpy as np

N = 20
L = 100.0
rho_steel = 4e-6 * .466e9
rho_brass = 4.5e-6 * .38e9
E_steel = 800.0e3
E_brass = 1500.0e3
rho = lambda x: rho_steel if x < 30 or x > 60 else rho_brass
E = lambda x: E_steel if x < 30 or x > 60 else E_brass
xpart = np.linspace(0, L, N + 1)
g = [300.0, None]
f_nodal = np.zeros(N + 1)
dt = .1
