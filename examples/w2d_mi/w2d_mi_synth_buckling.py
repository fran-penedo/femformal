import numpy as np

from femformal.core import logic, casestudy
from examples.w2d_mi.w2d_mi_synth_model import *

def sointerp(data):
    def shapes(x, coords):
        x1, x2, x3 = coords
        return np.array([(x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3)),
                         (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3)),
                         (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))])

    return lambda x: shapes(x, data[:, 0]).dot(data[:, 1])

def soderinterp(data):
    def shapes(x, coords):
        x1, x2, x3 = coords
        return np.array([((x - x2) + (x - x3)) / ((x1 - x2) * (x1 - x3)),
                         ((x - x1) + (x - x3)) / ((x2 - x1) * (x2 - x3)),
                         ((x - x1) + (x - x2)) / ((x3 - x1) * (x3 - x2))])

    return lambda x: shapes(x, data[:, 0]).dot(data[:, 1])


apc1_nodes = 1e3 * np.array([[0, -.15], [8, .1], [16, -.35]])
apc1 = logic.APCont2D(1, np.array([[2e3, 0], [14e3, 0]]), '>',
                      lambda x, y: sointerp(apc1_nodes)(x),
                      lambda x, y: abs(soderinterp(apc1_nodes)(x)))
apc2_nodes = 1e3 * np.array([[0, .2], [8, .35], [16, -.0]])
apc2 = logic.APCont2D(1, np.array([[2e3, 0], [14e3, 0]]), '<',
                      lambda x, y: sointerp(apc2_nodes)(x),
                      lambda x, y: abs(soderinterp(apc2_nodes)(x)))

cregions = {'A': apc1, 'B': apc2}
cspec = "((G_[3.00, 4.05] (A)) & (G_[3.00, 4.05] (B)))"
bounds = [-10e3, 10e3]

cs = casestudy.build_cs(
    sosys, [d0, v0], g, cregions, cspec,
    discretize_system=False, bounds=bounds, error_bounds=error_bounds,
    pset=[dset, vset, fset], f=[fd, fv, traction_force], system_t=sosys_t,
    d0_t=[d0_t, v0_t])
