from __future__ import division, absolute_import, print_function

import numpy as np

from femformal.core import system as sys
from femformal.core.fem import mechnlfem as mechnlfem
from examples.mm_nl2.mmnl_model import *

# from examples.mm_nl.results import mm_mxd_results as mdiff


d_par = 0.0
v_par = 0.0
dset = np.array([[1, d_par], [-1, d_par]])
vset = np.array([[1, v_par], [-1, v_par]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]

d0, v0 = mechnlfem.state(u0, du0, xpart, g)

input_dt = 0.05
ys = [0.0, 100.0, 100.0, -10, -10, -10, -10, -10, -10, -10, -10]
pwlf = sys.PWLFunction(
    np.linspace(0, T, round(T / input_dt) + 1), ys=None, ybounds=[-100.0, 100.0], x=L
)
fset = pwlf.pset()
fset[0, -1] = fset[fset.shape[0] // 2, -1] = 0.0

bigN_acc = np.max([bigN_int_force, np.max(np.abs(pwlf.ybounds))]) / (rho * L / N)
bigN_acc = 1000000

sosys = mechnlfem.mechnlfem(
    xpart, rho, E, g, f_nodal, dt, bigN_deltas, bigN_int_force, bigN_acc
)

# error_bounds = [[mdiff.eps, mdiff.eps_xderiv], [mdiff.eta, None], [mdiff.nu, mdiff.nu_xderiv]]
error_bounds = None
