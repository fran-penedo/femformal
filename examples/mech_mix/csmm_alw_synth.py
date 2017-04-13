import core.fem.mechlinfem as mechlinfem
import core.fem.fem_util as fem
import core.system as sys
import core.logic as logic
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
g = [0.0, None]
f_nodal = np.zeros(N + 1)
# f_nodal[-1] = 2e6
dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_steel / rho_steel))
fdt_mult = 1
bounds = [-10, 10]
# u0 = lambda x: 0.0
# du0 = lambda x: 0.0

apc1 = logic.APCont([35000, 55000], ">", lambda x: 4 * x / 100000.0 - 1.7 , lambda x: 4 / 100000.0)
apc2 = logic.APCont([10000, 25000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 2 / 100000.0)
apc3 = logic.APCont([65000, 90000], ">", lambda x: 2 * x / 100000.0 - 0.3 , lambda x: 2 / 100000.0)
# apc1 = logic.APCont([35000, 55000], ">", lambda x: 4 * x / 100000.0 - 1.0 , lambda x: 0.0)
# apc2 = logic.APCont([10000, 25000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 0.0)
# apc3 = logic.APCont([65000, 90000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
cspec = "G_[0.1, 0.3] ((A) & (C))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

eps = 0.41996649117561369
eta = [ 0.44282392,  0.43811652,  0.44146346,  0.4132631 ,  0.42811921,
        0.44102599,  0.8535847 ,  0.84274204,  0.78708181,  0.77689222,
        0.73910428,  0.71189787,  0.34288324,  0.37940241,  0.39175066,
        0.37327922,  0.38497219,  0.35811857,  0.35766271,  0.32906913]
nu = [ 0.        ,  0.08949916,  0.14018853,  0.14444168,  0.16910246,
        0.18570465,  0.20603731,  0.25935461,  0.31238539,  0.37661656,
        0.3993374 ,  0.45514495,  0.49579084,  0.52424749,  0.57550987,
        0.56982821,  0.60036769,  0.60721265,  0.62372119,  0.68169747,
        0.74884722]

dset = np.array([[1, 0], [-1, 0]])
vset = np.array([[1, 0], [-1, 0]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]
pwlf = sys.PWLFunction([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30], ybounds=[0e3, 5e3], x=L)
fset = pwlf.pset()


sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
# d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, None, g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, eps=eps, eta=eta, nu=nu)
cs.dsystem = cs.system

# print sosys
# print d0
# print v0
print cs.spec
# print dt

# import matplotlib.pyplot as plt
# sys.draw_sosys(sosys, d0, v0, g, 0.3, animate=False, hold=True)
# ax = plt.gcf().get_axes()[1]
# for apc in cregions.values():
#     ax.plot(apc.A, [apc.p(x) for x in apc.A], 'b-', lw=1)
# plt.show()

