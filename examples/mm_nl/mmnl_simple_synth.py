from __future__ import division, absolute_import, print_function

import numpy as np

from femformal.core import logic as logic, casestudy as fem
from examples.mm_nl.mmnl_synth_model import *

# apc1 = logic.APCont([45000, 60000], ">", lambda x: 1 * x / 100000.0 - 0.3 , lambda x: 1.0 / 100000.0)
# apc2 = logic.APCont([60000, 90000], ">", lambda x: 0.5 * x / 100000.0 + .3 , lambda x: 0.5 / 100000.0)
# apc3 = logic.APCont([60000, 90000], "<", lambda x: 0 * x / 100000.0 - 0.0 , lambda x: 0.0)
# cregions = {'B': apc2, 'C': apc3}
#
# cspec = "(G_[0.4, 0.5] (B))"
cregions = None
cspec = None


fdt_mult = 1
bounds = [-100, 100]

cs = fem.build_cs(sosys, [d0, v0], g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=error_bounds, T=T)
