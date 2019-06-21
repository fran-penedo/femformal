import numpy as np

from femformal.core import logic as logic, casestudy as fem
from examples.mech_mix.mm_synth_model import *

apc1 = logic.APCont([45000, 60000], ">", lambda x: 1 * x / 100000.0 - 0.3 , lambda x: 1.0 / 100000.0)
apc2 = logic.APCont([60000, 90000], ">", lambda x: 0.5 * x / 100000.0 + .3 , lambda x: 0.5 / 100000.0)
apc3 = logic.APCont([60000, 90000], "<", lambda x: 0 * x / 100000.0 - 0.0 , lambda x: 0.0)
cregions = {'B': apc2, 'C': apc3}

cspec = ("((G_[0.1, 0.2] (B)) & (F_[0.2, 0.4] (C)) & "
         "((G_[0.45, 0.5] (C) | G_[0.45, 0.5] (B))) & (F_[0.5, 0.55] (B)))")

fdt_mult = 1
bounds = [-100, 100]

cs = fem.build_cs(sosys, [d0, v0], g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, error_bounds=error_bounds)

# inputs = [0.0, 1987.9180545925, 2000.0, -2990.9140641370377, 3301.80750340456, 3879.863483563019]
# pwlf.ys = inputs
# def f_nodal_control(t):
#     f = np.zeros(N + 1)
#     f[-1] = pwlf(t, x=pwlf.x)
#     return f
# csys = sys.make_control_system(cs.system, f_nodal_control)
# cs.rob_tree = logic.csystem_robustness(cs.spec, csys, cs.d0, tree=True)
# pwlf.ys = None
