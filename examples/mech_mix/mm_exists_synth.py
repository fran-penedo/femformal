import numpy as np

from femformal.core import logic as logic, casestudy as fem
from examples.mech_mix.mm_synth_model import *

apc2 = logic.APCont([50000, 70000], ">", lambda x: 2.0 * x / 100000.0 + .0 , lambda x: 2.0 / 100000.0, quantifier=logic.Quantifier.exists)
apc3 = logic.APCont([60000, 90000], "<", lambda x: 0 * x / 100000.0 + 2.75 , lambda x: 0.0)
cregions = {'B': apc2, 'C': apc3}

cspec = ("((G_[0.2, 0.3] (B)) & (G_[0.0, 0.3] (C)))")

fdt_mult = 1
bounds = [-100, 100]

cs = fem.build_cs(sosys, [d0, v0], g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=error_bounds)
