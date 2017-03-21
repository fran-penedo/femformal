import fem.mechlinfem as mechlinfem
import fem.fem_util as fem
import femformal.system as sys
import femformal.logic as logic
import numpy as np

N = 50
L = 10.0
rho = 1.0
E = 1.0
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
f_nodal[-1] = 50
dt = .1
# u0 = lambda x: 0.0
# du0 = lambda x: 0.0

apc1 = logic.APCont([2, 8], -1, lambda x: 300.0 / 6.0 * (x - 2.0), lambda x: 300.0 / 6.0)
apc2 = logic.APCont([2, 8], 1, lambda x: 100 + 500.0 / 6.0 * (x - 2.0), lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2}

cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
# cspec = "(G_[9, 10] (B))"

dset = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 20.0], [0, -1, 0]])
vset = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 10.0], [0, -1, 0]])
fd = lambda x, p: p[0] * x + p[1]
fv = lambda x, p: p[0] * x + p[1]

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
cs = fem.build_cs(sosys, None, g, cregions, cspec,
                  discretize_system=False, pset=[dset, vset], f=[fd,fv])
cs.thunk = {'dt': cs.dt}
cs.dsystem = cs.system

# dsystem = cs.system
# d0 = cs.d0
# spec = cs.spec
# rh_N = cs.rh_N
# thunk = {'dt': cs.dt}

