from __future__ import division, absolute_import, print_function

import numpy as np

from femformal.core import system as sys
from femformal.core.fem import mechlinfem as mechlinfem
from examples.mech_mix.mm_model import *
from examples.mech_mix.results import mm_mxd_f_t0025_results as mdiff


d_par = 0.0
v_par = 0.0
dset = np.array([[1, d_par], [-1, d_par]])
vset = np.array([[1, v_par], [-1, v_par]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]

d0, v0 = mechlinfem.state(u0, du0, xpart, g)

input_dt = .1
pwlf = sys.PWLFunction(np.linspace(0, T, round(T / input_dt) + 1), ybounds=[-5e3, 5e3], x=L)
fset = pwlf.pset()
fset[0, -1] = fset[fset.shape[0] // 2, -1] = 0.0

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)

error_bounds = [[mdiff.eps, mdiff.eps_xderiv], [mdiff.eta, None], [mdiff.nu, mdiff.nu_xderiv]]
