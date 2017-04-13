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
fdt_mult = 2
bounds = [-10, 10]
# u0 = lambda x: 0.0
# du0 = lambda x: 0.0

apc1 = logic.APCont([30000, 60000], ">", lambda x: 4 * x / 100000.0 - 1.7 , lambda x: 4 / 100000.0)
apc2 = logic.APCont([60000, 90000], ">", lambda x: 2 * x / 100000.0 - 0.3 , lambda x: 2 / 100000.0)
apc3 = logic.APCont([30000, 60000], "<", lambda x: 0 * x / 100000.0 - .5 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
cspec = "((G_[0.1, 0.3] ((A) & (B))) & (F_[0.3, 0.4] (C)) & ((G_[0.4, 0.5] (C) | G_[0.4, 0.5] (A))) & (F_[0.5, 0.52] (A)))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

eps = 0.53060603311014809
eta = [ 0.49935304,  0.49260625,  0.48246315,  0.47431651,  0.4654891 ,
        0.4560288 ,  0.89437048,  0.86818276,  0.83171311,  0.78002793,
        0.72940515,  0.67932572,  0.3468329 ,  0.3985248 ,  0.41084645,
        0.39203747,  0.40435913,  0.37609546,  0.37559482,  0.33658952]
nu = [ 0.        ,  0.09105625,  0.14194901,  0.14760641,  0.17961008,
        0.18414384,  0.21788906,  0.2646227 ,  0.32974938,  0.39962464,
        0.42935625,  0.49479074,  0.54288331,  0.56175749,  0.59419818,
        0.58334308,  0.61170429,  0.6661196 ,  0.68726251,  0.74925849,
        0.81796891]

dset = np.array([[1, 0], [-1, 0]])
vset = np.array([[1, 0], [-1, 0]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]
pwlf = sys.PWLFunction(np.linspace(0, 0.55, 55/5 + 1), ybounds=[-5e3, 5e3], x=L)
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

