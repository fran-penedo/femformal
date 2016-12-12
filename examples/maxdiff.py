import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np


Ns = [10, 20, 30, 40, 50]
Nlen = len(Ns)
L = 10.0
T = [10.0, 100.0]
dt = .1

cstrue = fem.build_cs(
    1000, L, T, 0.01,
    None, None, None, discretize_system=False)

cslist = [fem.build_cs(N, L, T, dt, None, None, None) for N in Ns]

t0, tt = 1, 10
xl = 1.0
xr = 9.0
mds = [fem.max_diff(cs.system, cs.dt, cs.xpart, t0, tt, xl, xr, cs.T, cstrue)
       for cs in cslist]

print mds
