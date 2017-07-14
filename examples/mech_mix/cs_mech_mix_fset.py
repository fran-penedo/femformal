import numpy as np

from femformal.core import logic as logic, fem_util as fem
from femformal.core.fem import mechlinfem as mechlinfem


N = 20
L = 100000
rho_steel = 8e-6
rho_brass = 8.5e-6
E_steel = 200e6
E_brass = 100e6
rho = lambda x: rho_steel if x < 30000 or x > 60000 else rho_brass
E = lambda x: E_steel if x < 30000 or x > 60000 else E_brass
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
# f_nodal[-1] = 2e6
dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_steel / rho_steel))
# u0 = lambda x: 0.0
# du0 = lambda x: 0.0

apc1 = logic.APCont([35000, 55000], 1, lambda x: 4 * x / 100000.0 + 1.5 , lambda x: 0.0)
apc2 = logic.APCont([10000, 25000], 1, lambda x: 2 * x / 100000.0 + 1.0 , lambda x: 0.0)
apc3 = logic.APCont([65000, 90000], 1, lambda x: 2 * x / 100000.0 + 1.0 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
cspec = "G_[0.0, 0.3] ((A) & (B) & (C))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

eps = 0.46369130943083958

dset = np.array([[1, 0], [-1, 0]])
vset = np.array([[1, 0], [-1, 0]])
fset = np.array([[-1, -1.9e3], [1, 2.1e3]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]
ff = lambda x, t, p: 0.0 if x < L else p[0]


sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
# d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, None, g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, ff],
                  eps=eps)
cs.dsystem = cs.system

# print sosys
# print d0
# print v0
# print cs.spec
# print dt

# import matplotlib.pyplot as plt
# sys.draw_sosys(sosys, d0, v0, g, 0.3, animate=False, hold=True)
# ax = plt.gcf().get_axes()[1]
# for apc in cregions.values():
#     ax.plot(apc.A, [apc.p(x) for x in apc.A], 'b-', lw=1)
# plt.show()

