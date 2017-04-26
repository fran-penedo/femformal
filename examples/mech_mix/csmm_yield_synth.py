import femformal.core.fem.mechlinfem as mechlinfem
import femformal.core.fem.fem_util as fem
import femformal.core.system as sys
import femformal.core.logic as logic
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

apc1 = logic.APCont([30000, 60000], "<", lambda x: 4e-5 , lambda x: 0.0)
apc2 = logic.APCont([30000, 60000], ">", lambda x: 2e-5, lambda x: 0.0)
apc3 = logic.APCont([30000, 35000], ">", lambda x: 4e-5 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
cspec = "((G_[0, 0.3] (A)) & (G_[0.1, 0.3] (B)) & (F_[0.3, 0.5] (C)))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

eps = [ 0.09747385, 0.14081154, 0.17891219, 0.23712687, 0.2631131,  0.28320132,
  0.34506635, 0.44464238, 0.52170038, 0.57311389, 0.63457094, 0.73045952,
  0.76688038, 0.79295413, 0.81073248, 0.80988182, 0.80980781, 0.81635747,
  0.84126566, 0.88176249]
eps_xderiv = [  3.63710469e-05,  2.11281198e-05,  2.45120308e-05,  2.12092292e-05,
   2.29235165e-05,  8.81787146e-05,  8.81787146e-05,  4.58199506e-05,
   3.81767913e-05,  4.05394899e-05,  3.74659992e-05,  6.22872747e-05,
   6.19170237e-05,  5.63736186e-05,  7.13999328e-05,  6.23714894e-05,
   6.36109354e-05,  6.91969134e-05,  6.42082753e-05]
eta = [ 0.8817108,  0.86919647, 0.85573683, 0.85692228, 0.85712663, 0.84467017,
  1.66180179, 1.61187177, 1.51651829, 1.41832172, 1.32893972, 1.20596396,
  0.57127342, 0.53553381, 0.54071544, 0.46455876, 0.42099359, 0.39978828,
  0.36893354, 0.33233276]
nu = [ 0.,          0.08975913, 0.14445103, 0.21245778, 0.27922077, 0.32448564,
  0.38610786, 0.49358133, 0.60423934, 0.72094741, 0.81646251, 0.91930143,
  1.01201858, 1.05056047, 1.0770801,  1.09663198, 1.13879711, 1.14350707,
  1.18311543, 1.20747518, 1.23564469]
nu_xderiv = [  1.79705700e-05,  1.20243419e-05,  1.52513759e-05,  1.50174423e-05,
   1.27364425e-05,  1.43272903e-05,  2.62368572e-05,  2.48319881e-05,
   2.66662141e-05,  2.73318215e-05,  2.91345263e-05,  4.82373530e-05,
   5.14398040e-05,  6.53855933e-05,  8.01887067e-05,  7.08245400e-05,
   7.32814442e-05,  7.97948303e-05,  6.57721218e-05,  8.00374492e-05]

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

