import numpy as np
from matplotlib import pyplot as plt

from femformal.core import system as sys, logic as logic, casestudy as fem
from femformal.core.fem import mechlinfem as mechlinfem


N = 20
L = 100.0
# rho = .000724
# E = 30e6
rho = 8e3
E = 200e9
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
f_nodal[-1] = 2e6
dt = 5.0 / np.sqrt(E / rho)
u0 = lambda x: 0.0
du0 = lambda x: 0.0

apc1 = logic.APCont([5, 25], 1, lambda x: 3e-3 * x / 100 , lambda x: 0.0)
apc2 = logic.APCont([5, 25], -1, lambda x: 7e-3 * x / 100 + 1.0e-3, lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
cspec = "G_[0.001, 0.005] (F_[0.0, 0.002] (A) & F_[0.0, 0.002] (B))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, [d0, v0], g, cregions, cspec, discretize_system=False)

print sosys
print d0
print v0
print cs.spec
print dt
sys._draw_sosys(sosys, d0, v0, g, 0.2, animate=False, hold=True)
# ax = plt.gcf().get_axes()[1]
# for apc in [apc1, apc2]:
#     ax.plot(apc.A, [apc.p(x) for x in apc.A], 'b-', lw=1)
plt.show()

dsystem = cs.system
d0 = cs.d0
spec = cs.spec
rh_N = cs.rh_N
thunk = {'dt': cs.dt}

