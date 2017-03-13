import fem.mechlinfem as mechlinfem
import fem.fem_util as fem
import femformal.system as sys
import femformal.logic as logic
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
f_nodal[-1] = 2e3
dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_steel / rho_steel))
u0 = lambda x: 0.0
du0 = lambda x: 0.0

apc1 = logic.APCont([35000, 55000], 1, lambda x: 4 * x / 100000.0 + 1 , lambda x: 0.0)
apc2 = logic.APCont([10000, 25000], 1, lambda x: 2 * x / 100000.0 + .5 , lambda x: 0.0)
apc3 = logic.APCont([65000, 90000], 1, lambda x: 2 * x / 100000.0 + .5 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

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

import matplotlib.pyplot as plt
sys.draw_sosys(sosys, d0, v0, g, 0.3, animate=False, hold=True)
ax = plt.gcf().get_axes()[1]
for apc in cregions.values():
    ax.plot(apc.A, [apc.p(x) for x in apc.A], 'b-', lw=1)
plt.show()

dsystem = cs.system
d0 = cs.d0
spec = cs.spec
rh_N = cs.rh_N
thunk = {'dt': cs.dt}

