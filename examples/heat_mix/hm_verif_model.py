import numpy as np

from femformal.core import system as sys
from femformal.core.fem import heatlinfem as heatlinfem
from examples.heat_mix.batch_hm_model import *
from examples.heat_mix.results import hm_maxdiff_results_N30 as mdiff
from examples.heat_mix.resutls.hm_synth_simple2_N30_results import inputs

d_par = 300.0
dset = np.array([[1, d_par], [-1, -d_par]])
fd = lambda x, p: p[0]
u0 = lambda x: fd(x, [d_par])
d0 = heatlinfem.state(u0, xpart, g)
T = 5.0
input_dt = 0.5
# inputs = [232941.6924911008, 1000000.0, 1000000.0, 1000000.0, 792753.9815042098, 635825.568434383, 566318.4181989289, 91957.27671086021, 0.0, 377290.0006792998, 0.0]
pwlf = sys.PWLFunction(np.linspace(0, T, round(T / input_dt) + 1), ys=inputs, x=L)
fset = pwlf.pset()
fosys = heatlinfem.heatlinfem_mix(xpart, rho, E, g, f_nodal, dt)

error_bounds = [[None, None], [None, None], [None, None]]
