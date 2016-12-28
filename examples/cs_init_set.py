import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np


N = 50
L = 10.0
T = [10.0, 100.0]
dt = .1

apc1 = logic.APCont([1, 9], -1, lambda x: 9.0 * x)
apc2 = logic.APCont([6, 7], 1, lambda x: 9.0 * x + 15.0)
cregions = {'A': apc1, 'B': apc2}

cspec = "((G_[1, 10] (A)) & (F_[4, 6] (B)))"
# t \in [1,10], T = [10, 100], x \in [1, 9], N = [10, 20, 30, 40, 50], L = 10
eps = 1.0

pset = np.array([[1, 0, 9], [-1, 0, -9], [0, 1, 10.0], [0, -1, -0.0]])

cs = fem.build_cs(N, L, T, dt, None, cregions, cspec, eps=eps, pset=pset)

print "loaded cs"


# a = 9, 5 < b < 15
# Res: 4.00501710467
# Time: 4291.1989789

# a = 9, 0 < b < 10
# Res: -0.989965788443
# Time: 315.550029993
