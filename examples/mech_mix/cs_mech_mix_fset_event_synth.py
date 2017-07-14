import numpy as np

from femformal.core import system as sys, logic as logic
from femformal.core.fem import mechlinfem as mechlinfem, fem_util as fem


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
fdt_mult = 3
# u0 = lambda x: 0.0
# du0 = lambda x: 0.0

# apc1 = logic.APCont([35000, 55000], ">", lambda x: 4 * x / 100000.0 - 1.0 , lambda x: 4 / 100000.0)
# apc2 = logic.APCont([10000, 25000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 2 / 100000.0)
# apc3 = logic.APCont([65000, 90000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 2 / 100000.0)
apc1 = logic.APCont([35000, 55000], ">", lambda x: 4 * x / 100000.0 - 1.0 , lambda x: 0.0)
apc2 = logic.APCont([10000, 25000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 0.0)
apc3 = logic.APCont([65000, 90000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
cspec = "F_[0.0, 0.3] ((A) & (C))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

eps = 0.46369130943083958
eta = [ 0.13755909,  0.13126048,  0.13162718,  0.1222216 ,  0.13138857,
        0.84161904,  0.26558981,  0.27607333,  0.2477102 ,  0.25094865,
        0.26652575,  0.26906774,  1.22645221,  0.1283975 ,  0.12622162,
        0.11715973,  0.12185615,  0.11242183,  0.10054052,  0.08407562]
nu = [ 0.        ,  0.03420027,  0.05542994,  0.06004345,  0.06031116,
        0.06031116,  0.13057483,  0.12971402,  0.125502  ,  0.13602502,
        0.1138558 ,  0.10627603,  0.13725949,  0.08015514,  0.0780944 ,
        0.07784882,  0.07825857,  0.07887112,  0.08361303,  0.08610627,
        0.10080335]

dset = np.array([[1, 0], [-1, 0]])
vset = np.array([[1, 0], [-1, 0]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]
pwlf = sys.PWLFunction([0.0, 0.1, 0.2, 0.3], ybounds=[2e3, 3e3], x=L)
fset = pwlf.pset()


sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
# d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, None, g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, pwlf], fdt_mult=fdt_mult,
                  eps=eps, eta=eta, nu=eta)
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

