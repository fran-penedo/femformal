import numpy as np

from femformal.core import system as sys, logic as logic, casestudy as fem
from femformal.core.fem import heatlinfem as heatlinfem
from examples.heat_mix.hm_model import *
from examples.heat_mix.results import hm_maxdiff_results as mdiff

fdt_mult = 2
bounds = [-1e4, 1e4]

d_par = 300.0
dset = np.array([[1, d_par], [-1, d_par]])
fd = lambda x, p: p[0]
u0 = lambda x: fd(x, [d_par])
d0 = heatlinfem.state(u0, xpart, g)
T = 5.0
input_dt = 0.5
pwlf = sys.PWLFunction(np.linspace(0, T, round(T / input_dt) + 1), ybounds=[0, 1e6], x=L)
fset = pwlf.pset()

apc1 = logic.APCont([30, 60], ">", lambda x: .5 * x + 290, lambda x: 1.0)
apc2 = logic.APCont([30, 60], "<", lambda x: .5 * x + 310, lambda x: 1.0)
cregions = {'A': apc1, 'B': apc2}

cspec = "(G_[4.5, 5.0] ((A) & (B)))"

error_bounds = [[mdiff.eps, None], [mdiff.eta, None], [mdiff.nu, None]]

fosys = heatlinfem.heatlinfem_mix(xpart, rho, E, g, f_nodal, dt)
cs = fem.build_cs(fosys, d0, g, cregions, cspec, discretize_system=False,
                  pset=[dset, fset], f=[fd, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=error_bounds, T=T)
cs.dsystem = cs.system
