import numpy as np

from femformal.core import system as sys
from femformal.core.fem import mech2d as mech2d
from examples.column2d.c2d_model import *
from examples.column2d.results import c2d_maxdiff_results as mdiff


T = 5.0

sosys = mech2d.mech2d(xs, ys, rho, C, g, force, dt, None, q4=False)

input_dt = 0.75
pwlf = sys.PWLFunction(
    np.linspace(0, T, round(T / input_dt) + 1), ybounds=[-4e3, 0.0], x=None
)
traction_force = mech2d.TimeVaryingTractionForce(pwlf, traction_templ, sosys.mesh)

d_par = 0.0
v_par = 0.0
dset = np.array([[1, d_par], [-1, d_par]])
vset = np.array([[1, v_par], [-1, v_par]])
fset = pwlf.pset()
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]

d0, v0 = mech2d.state(u0, du0, sosys.mesh.nodes_coords, g)

sosys_t = sys.ControlSOSystem.from_sosys(
    mech2d.mech2d(xs_t, ys_t, rho, C, g, force_t, dt_t, q4=False), None
)
traction_force_t = mech2d.TimeVaryingTractionForce(pwlf, traction_templ, sosys_t.mesh)
sosys_t.add_f_nodal(traction_force_t.traction_force)
d0_t, v0_t = mech2d.state(u0, du0, sosys_t.mesh.nodes_coords, g)

error_bounds = [[mdiff.eps, None], [mdiff.eta, None], [mdiff.nu, None]]
# error_bounds = None
