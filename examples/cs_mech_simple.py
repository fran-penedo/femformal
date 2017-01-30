import fem.mechlinfem as mechlinfem
import fem.fem_util as fem
import femformal.system as sys
import femformal.logic as logic
import numpy as np

N = 50
L = 10.0
rho = 1.0
E = 1.0
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
f_nodal[-1] = 50
dt = .1
u0 = lambda x: 50.0
du0 = lambda x: 10.0

apc1 = logic.APCont([2, 8], -1, lambda x: 300.0 / 6.0 * (x - 2.0), lambda x: 300.0 / 6.0)
apc2 = logic.APCont([2, 8], 1, lambda x: 100 + 500.0 / 6.0 * (x - 2.0), lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
cspec = "(G_[9, 10] (B))"

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal)
d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, xpart, dt, [d0, v0], g, cregions, cspec, discretize_system=False)

print sosys
print d0
print v0

import matplotlib.pyplot as plt
sys.draw_sosys(sosys, d0, v0, g, 10, xpart, animate=False, hold=True)
ax = plt.gcf().get_axes()[1]
for apc in [apc1, apc2]:
    ax.plot(apc.A, [apc.p(x) for x in apc.A], 'b-', lw=1)
plt.show()

dsystem = cs.system
d0 = cs.d0
spec = cs.spec
rh_N = cs.rh_N
thunk = {'dt': cs.dt}

