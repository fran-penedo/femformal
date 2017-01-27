import fem.mechlinfem as mechlinfem
import fem.fem_util as fem
import femformal.system as sys
import femformal.logic as logic
import numpy as np

N = 10
L = 10.0
rho = 1.0
E = 1.0
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
f_nodal[-1] = 50
dt = .1
u0 = lambda x: 0.0
du0 = lambda x: 0.0

apc1 = logic.APCont([2, 8], -1, lambda x: 300 / 6 * (x - 2.0), lambda x: 300 / 6)
cregions = {'A': apc1}

cspec = "(F_[1, 10] (A))"

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal)
d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, xpart, dt, [d0, v0], g, cregions, cspec, discretize_system=False)

print sosys
print d0
print v0

sys.draw_sosys(sosys, d0, v0, g, 10, xpart, animate=False)
