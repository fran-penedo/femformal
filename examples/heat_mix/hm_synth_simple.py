import numpy as np

from femformal.core import logic, casestudy as fem
from examples.heat_mix.hm_synth_model import *

apc1 = logic.APCont([30, 60], ">", lambda x: .25 * x + 298, lambda x: .25)
apc2 = logic.APCont([30, 60], "<", lambda x: .25 * x + 302, lambda x: .25)
cregions = {'A': apc1, 'B': apc2}

cspec = "(G_[4.0, 5.0] ((A) & (B)))"

fdt_mult = 2
bounds = [-1e4, 1e4]

cs = fem.build_cs(fosys, d0, g, cregions, cspec, discretize_system=False,
                  pset=[dset, fset], f=[fd, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=error_bounds, T=T)
