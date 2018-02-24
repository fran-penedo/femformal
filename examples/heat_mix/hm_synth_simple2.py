import numpy as np

from femformal.core import logic, casestudy as fem
from examples.heat_mix.hm_synth_model import *

apc1 = logic.APCont([30, 60], ">", lambda x: .25 * x + 297, lambda x: .25)
apc2 = logic.APCont([30, 60], "<", lambda x: .25 * x + 303, lambda x: .25)
apc3 = logic.APCont([100, 100], "<", lambda x: 345.0, lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

cspec = "((G_[4.0, 5.0] ((A) & (B))) & (G_[0.0, 5.0] (C)))"

fdt_mult = 1
bounds = [-1e4, 1e4]

cs = fem.build_cs(fosys, d0, g, cregions, cspec, discretize_system=False,
                  pset=[dset, fset], f=[fd, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=error_bounds, T=T)
