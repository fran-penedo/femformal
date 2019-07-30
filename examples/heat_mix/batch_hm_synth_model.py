import numpy as np

from femformal.core import system as sys
from femformal.core.fem import heatlinfem as heatlinfem
from examples.heat_mix.batch_hm_model import *
from examples.heat_mix.results import hm_maxdiff_results_N100 as mdiff

d_par = 300.0
dset = np.array([[1, d_par], [-1, -d_par]])
fd = lambda x, p: p[0]
u0 = lambda x: fd(x, [d_par])
d0 = heatlinfem.state(u0, xpart, g)
T = 5.0
input_dt = 0.5
pwlf = sys.PWLFunction(
    np.linspace(0, T, round(T / input_dt) + 1), ybounds=[0.0, 1e6], x=L
)
fset = pwlf.pset()
fosys = heatlinfem.heatlinfem_mix(xpart, rho, E, g, f_nodal, dt)

error_bounds = [[mdiff.eps, None], [mdiff.eta, None], [mdiff.nu, None]]
