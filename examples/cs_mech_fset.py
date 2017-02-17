import fem.mechlinfem as mechlinfem
import fem.fem_util as fem
import femformal.system as sys
import femformal.logic as logic
import numpy as np

N = 20
L = 100.0
rho = .000724
E = 30e6
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
# f_nodal[-1] = 50
dt = 5.0 / np.sqrt(E / rho)
# u0 = lambda x: 0.0
# du0 = lambda x: 0.0

apc1 = logic.APCont([20, 80], 1, lambda x: 3e-3 * x / 100 , lambda x: 0.0)
apc2 = logic.APCont([20, 80], -1, lambda x: 7e-3 * x / 100 - .5e-3, lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2}

cspec = "G_[0.001, 0.0015] (A)"

dset = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 20.0], [0, -1, 0]])
vset = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 10.0], [0, -1, 0]])
fset = np.array([[-1, 900.0], [1, 1100.0]])
fd = lambda x, p: p[0] * x + p[1]
fv = lambda x, p: p[0] * x + p[1]
ff = lambda x, p: 0 if x < L else p[0]

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
cs = fem.build_cs(sosys, None, g, cregions, cspec,
                  discretize_system=False, pset=[dset, vset, fset], f=[fd,fv,ff])
cs.dsystem = cs.system

# dsystem = cs.system
# d0 = cs.d0
# spec = cs.spec
# rh_N = cs.rh_N
# thunk = {'dt': cs.dt}

