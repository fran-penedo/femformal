import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np


N = 50
L = 10.0
T = [10.0, 100.0]
dt = .1

# d0s = [[T[0]] + [min(max(a * x + b, T[0]), T[1])
#                  for x in np.linspace(0, L, N + 1)[1:-1]] + [T[1]]
#        for (a, b), N in zip(ablist, Ns)]
(a, b) = (9, 0)
d0 = [T[0]] + [a * x + b for x in np.linspace(0, L, N + 1)[1:-1]] + [T[1]]

apc1 = logic.APCont([1, 9], -1, lambda x: 9.0 * x)
apc2 = logic.APCont([6, 7], 1, lambda x: 9.0 * x + 15.0)
cregions = {'A': apc1, 'B': apc2}

cspec = "((G_[1, 10] (A)) & (F_[4, 6] (B)))"
# t \in [1,10], T = [10, 100], x \in [1, 9], N = [10, 20, 30, 40, 50], L = 10
eps = 1.0

cs = fem.build_cs(N, L, T, dt, d0, cregions, cspec, eps=eps)
system = cs.system
rh_N = cs.rh_N
spec = cs.spec

import femformal.system as fsys

fsys.draw_system_disc(system, d0, dt, 10, cs.xpart, animate=False)


print "loaded cs"
