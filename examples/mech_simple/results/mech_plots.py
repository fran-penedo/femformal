import numpy as np
from matplotlib import pyplot as plt

from femformal.core import system as sys, logic as logic
from femformal.core.fem import mechlinfem as mechlinfem


N = 20
L = 100.0
rho = .000724
E = 30e6
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
f_nodal[-1] = 900
dt = 5.0 / np.sqrt(E / rho)
u0 = lambda x: 0.0
du0 = lambda x: 0.0

apc1 = logic.APCont([20, 80], 1, lambda x: 3e-3 * x / 100 , lambda x: 3e-5)
apc2 = logic.APCont([20, 80], 1, lambda x: 7e-3 * x / 100 - 1.5e-3, lambda x: 7e-5)
cregions = {'A': apc1, 'B': apc2}

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
d0, v0 = mechlinfem.state(u0, du0, xpart, g)
# cs = fem.build_cs(sosys, [d0, v0], g, cregions, None, discretize_system=False)
sys.draw_sosys(sosys, d0, v0, g, 0.01, animate=False, hold=True, allonly=True)
fig = plt.gcf()
ax = plt.gcf().get_axes()[0]
for apc, label, fmt in zip([apc1, apc2], ['$\mu_1$', '$\mu_2$'], ['b-', 'g-']):
    ax.plot(apc.A, [apc.p(x) for x in apc.A], fmt, lw=1, label=label)
ax.legend(loc='upper left')
# fig.savefig('mech_plots{}.png'.format(2))
plt.show()
