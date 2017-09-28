import numpy as np

from femformal.core import system as sys, logic as logic, casestudy as fem
from femformal.core.fem import heatlinfem as heatlinfem


N = 20
L = 100
rho_steel = 1.0
rho_brass = 1.5
E_steel = 100.0
E_brass = 200.0
rho = lambda x: rho_steel if x < 30 or x > 60 else rho_brass
E = lambda x: E_steel if x < 30 or x > 60 else E_brass
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
# f_nodal[-1] = 2e6
dt = .01
# dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_brass / rho_brass))
fdt_mult = 2
bounds = [-10, 10]

d_par = 0.0
dset = np.array([[1, d_par], [-1, d_par]])
fd = lambda x, p: p[0]
u0 = lambda x: fd(x, [d_par])
d0 = heatlinfem.state(u0, xpart, g)
T = 0.55
input_dt = 0.05
pwlf = sys.PWLFunction(np.linspace(0, T, round(T / input_dt) + 1), ybounds=[-5e3, 5e3], x=L)
fset = pwlf.pset()

# apc1 = logic.APCont([45000, 60000], ">", lambda x: 1 * x / 100000.0 - 0.3 , lambda x: 1.0 / 100000.0)
# apc2 = logic.APCont([60000, 90000], ">", lambda x: 0.5 * x / 100000.0 + 0.0 , lambda x: 0.5 / 100000.0)
# apc3 = logic.APCont([60000, 90000], "<", lambda x: 0 * x / 100000.0 - 0.0 , lambda x: 0.0)
# cregions = {'B': apc2, 'C': apc3}
#
# cspec = ("((G_[0.1, 0.2] (B)) & (F_[0.2, 0.4] (C)) & "
#          "((G_[0.45, 0.5] (C) | G_[0.45, 0.5] (B))) & (F_[0.5, 0.52] (B)))")

fosys = heatlinfem.heatlinfem_mix(xpart, rho, E, g, f_nodal, dt)
cs = fem.build_cs(fosys, d0, g, None, None, discretize_system=False,
                  pset=[dset, fset], f=[fd, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=None, T=T)
cs.dsystem = cs.system
