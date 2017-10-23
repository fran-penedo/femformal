import numpy as np

from femformal.core import logic, casestudy
from examples.column2d.c2d_synth_model import *
from examples.column2d.results import c2d_maxdiff_results as mdiff

v = .1
apc1 = logic.APCont2D(1, np.array([[0.0, 0], [16, 0]]), '>',
                      lambda x, y: v * (-x * x / 64.0 + x / 4.0),
                      lambda x, y: v * (-x / 32.0 + 1/4.0))
cregions = {'A': apc1}
cspec = "(G_[2.0, 2.5] (A))".format(T)
bounds = [-10e-0, 10e-0]

cs = casestudy.build_cs(
    sosys, [d0, v0], g, cregions, cspec,
    discretize_system=False, bounds=bounds, error_bounds=error_bounds,
    pset=[dset, vset, fset], f=[fd, fv, traction_force])
