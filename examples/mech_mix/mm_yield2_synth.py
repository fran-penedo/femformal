import numpy as np

from femformal.core import logic as logic, casestudy as fem
from examples.mech_mix.mm_synth_model import *


apc1 = logic.APCont([30000, 60000], "<", lambda x: 2e-5, lambda x: 0.0, uderivs=1)
apc2 = logic.APCont([30000, 60000], ">", lambda x: -3e-5, lambda x: 0.0, uderivs=1)
apc3 = logic.APCont([30000, 60000], ">", lambda x: 3e-5, lambda x: 0.0, uderivs=1)
cregions = {"A": apc1, "C": apc3}
cspec = "((G_[0.10, 0.4] (A)) & (F_[0.4, 0.5] (C)))"
# cspec = "(G_[0.10, 0.4] (C))"

fdt_mult = 1
bounds = [-1e-3, 1e-3]

cs = fem.build_cs(
    sosys,
    [d0, v0],
    g,
    cregions,
    cspec,
    discretize_system=False,
    pset=[dset, vset, fset],
    f=[fd, fv, pwlf],
    fdt_mult=fdt_mult,
    bounds=bounds,
    error_bounds=error_bounds,
)
