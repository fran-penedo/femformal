import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np
import femformal.system as sys


Ns = [10]
Nlen = len(Ns)
L = 10.0
T = [10.0, 100.0]
dt = .1
d0s = [[T[0]] + [20.0 for i in range(N - 1)] + [T[1]] for N in Ns]

apc1 = logic.APCont([1, L - 1], -1, lambda x: 10 + 9 * x)
apc2 = logic.APCont([L/2, L - 1], 1, lambda x: 125)
cregions = {'A': apc1, 'B': apc2}

cspec = "(G_[1, 10] (A))"
# t \in [1,10], T = [10, 100], x \in [1, 9], N = [10, 20, 30, 40, 50], L = 10
eps = [5.10]

cstrue = fem.build_cs(
    1000, L, T, 0.01, [T[0]] + [20.0 for i in range(1000 - 1)] + [T[1]],
    cregions, cspec, discretize_system=False)

cslist = [fem.build_cs(N, L, T, dt, d0, cregions, cspec, eps=e)
          for N, d0, e in zip(Ns, d0s, eps)]

cs = cslist[0]

sys.sys_diff(cs.system, cstrue.system, cs.dt, cstrue.dt, cs.xpart, cstrue.xpart,
             cs.d0, cstrue.d0, t0=1, T=10, xl=1, xr=9, plot=True)

print "loaded cs"
