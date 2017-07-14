import numpy as np

from femformal.core import logic as logic, casestudy as fem
from femformal.core.fem import mechlinfem as mechlinfem


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

apc1 = logic.APCont([20, 80], 1, lambda x: 3e-3 * x / 100 , lambda x: 3e-5)
apc2 = logic.APCont([20, 80], 1, lambda x: 7e-3 * x / 100 + 1.0e-3, lambda x: 7e-5)
cregions = {'A': apc1, 'B': apc2}

cspec = "G_[0.001, 0.005] (B)"

eps = 9.6884306133304676e-16

eta = [ 0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452,
        0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452,
        0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452,
        0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452]
nu = [ 0.        ,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226]

dset = np.array([[1, 0], [-1, 0]])
vset = np.array([[1, 0], [-1, 0]])
fset = np.array([[-1, -900.0], [1, 1100.0]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]
ff = lambda x, p: 0.0 if x < L else p[0]

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
cs = fem.build_cs(sosys, None, g, cregions, cspec,
                  discretize_system=False, pset=[dset, vset, fset], f=[fd,fv,ff],
                  eps=eps, eta=eta, nu=None)
cs.dsystem = cs.system
