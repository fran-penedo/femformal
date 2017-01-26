import mechlinfem
import fem.fem_util as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np

N = 50
L = 10.0
rho = 1.0
E = 1.0
xpart = np.linspace(0, L, N + 1)
g = [0, 0]
f_nodal = np.zeros((N - 1, 1))
f_nodal[-1] = 50
dt = .1
d0 = lambda x: 0.0
dd0 = lambda x: 0.0

apc1 = logic.APCont([8, 9], -1, lambda x: 32.0 + (60.0 - 32.0) * (x - 8.0), lambda x: 60.0 - 32.0)
apc2 = logic.APCont([L/2, L - 1], 1, lambda x: 125, lambda x: 0)
cregions = {'A': apc1, 'B': apc2}

cspec = "(G_[1, 10] (A))"

system = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal)
y0 = init_state(d0, dd0, xpart, g)
cs = fem.build_cs(system, xpart, dt, y0, g, cregions, cspec)
