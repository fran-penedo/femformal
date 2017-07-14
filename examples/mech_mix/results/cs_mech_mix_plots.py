import numpy as np
from matplotlib import pyplot as plt

from femformal.core import system as sys, logic as logic, casestudy as fem
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
f_nodal[-1] = 2e3
dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_steel / rho_steel))
u0 = lambda x: 0.0
du0 = lambda x: 0.0

# apc1 = logic.APCont([40000, 50000], 1, lambda x: 4 * x / 100000.0 - 0.7 , lambda x: 0.0)
# apc2 = logic.APCont([40000, 50000], 1, lambda x: 2 * x / 100000.0 + 0.0 , lambda x: 0.0)
# # apc2 = logic.APCont([10000, 25000], 1, lambda x: 2 * x / 100000.0 + .5 , lambda x: 0.0)
# apc3 = logic.APCont([65000, 90000], 1, lambda x: 2 * x / 100000.0 + .5 , lambda x: 0.0)
# apc1 = logic.APCont([35000, 55000], ">", lambda x: 4 * x / 100000.0 - 0.7 , lambda x: 0.0)
# apc2 = logic.APCont([10000, 25000], ">", lambda x: 2 * x / 100000.0 - 0.7 , lambda x: 0.0)
# apc3 = logic.APCont([65000, 90000], ">", lambda x: 2 * x / 100000.0 - 0.7 , lambda x: 0.0)
apc1 = logic.APCont([35000, 55000], ">", lambda x: 4 * x / 100000.0 - 1.0 , lambda x: 0.0)
# apc2 = logic.APCont([10000, 25000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 0.0)
apc3 = logic.APCont([65000, 90000], ">", lambda x: 2 * x / 100000.0 - 1.0 , lambda x: 0.0)
apc4 = logic.APCont([35000, 55000], "<", lambda x: 4 * x / 100000.0 + 1.5 , lambda x: 0.0)
apc5 = logic.APCont([65000, 90000], "<", lambda x: 2 * x / 100000.0 + 1.0 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc3, 'C': apc4, 'D': apc5}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
# cspec = "G_[0.001, 0.005] (F_[0.0, 0.002] (A) & F_[0.0, 0.002] (B))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, [d0, v0], g, cregions, None, discretize_system=False)

print sosys
print d0
print v0
print cs.spec
print dt
sys.draw_sosys(sosys, d0, v0, g, 0.1, animate=False, allonly=True, hold=True,
               ylabel='Displacement u (mm)', xlabel='Location x (m)')
fig = plt.gcf()
fig.set_size_inches(4,2)
ax = plt.gcf().get_axes()[0]
labels = ['$\mu_1(-1)$', '$\mu_2(-1)$', '$\mu_1(1.5)$', '$\mu_2(1)$']
for (key, apc), label in zip(sorted(cregions.items()), labels):
    print key, label
    ax.plot(apc.A, [apc.p(x) for x in apc.A], lw=1, label=label)
ax.autoscale()
ax.legend(loc='upper left', framealpha=0.5, fontsize='9', labelspacing=0.05, handletextpad=0.1)
ax.set_xticklabels([x / 1000 for x in ax.get_xticks()])
# plt.show()
fig.savefig('mech_plots.png')

dsystem = cs.system
d0 = cs.d0
spec = cs.spec
rh_N = cs.rh_N
thunk = {'dt': cs.dt}

