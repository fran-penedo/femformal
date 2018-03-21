from __future__ import division, absolute_import, print_function

from femformal.core import logic as logic, casestudy as fem
from examples.mech_mix.mm_synth_model import *

apc1 = logic.APCont([60000, 90000], ">", lambda x: 2 * x / 100000.0 - 1.1 , lambda x: 2 / 100000.0)
apc2 = logic.APCont([60000, 90000], "<", lambda x: -2 * x / 100000.0 + 1.1 , lambda x: 2 / 100000.0)
cregions = {'A': apc1, 'B': apc2}

cspec = "((G_[0.1, 0.3] (A)) & (G_[0.4, 0.5] (B)))"

fdt_mult = 2
bounds = [-10, 10]

cs = fem.build_cs(sosys, [d0, v0], g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=error_bounds)
cs.dsystem = cs.system
