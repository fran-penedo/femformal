import numpy as np

from femformal.core import system as sys, draw_util as draw, logic, casestudy
from femformal.core.fem import mech2d as mech2d
from examples.column2d.results import c2d_maxdiff_results as mdiff
from examples.column2d.c2d_model import *


sosys = mech2d.mech2d(xs, ys, rho, C, g, force, dt, None, q4=False)

F = -4000000.0
input_dt = 0.5
inputs = [0, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F]
pwlf = sys.PWLFunction(
    np.linspace(0, T, round(T / input_dt) + 1), ys=inputs, x=None)
traction_force = mech2d.TimeVaryingTractionForce(pwlf, traction_templ, sosys.mesh)

# control_sys = sys.ControlSOSystem.from_sosys(sosys, traction_force.traction_force)

d0, v0 = mech2d.state(u0, du0, sosys.mesh.nodes_coords, g)

v = .1
apc1 = logic.APCont2D(1, np.array([[0.0, 0], [16, 0]]), '<',
                      lambda x, y: v * (-x * x / 64.0 + x / 4.0),
                      lambda x, y: v * (-x / 32.0 + 1/4.0))
cregions = {'A': apc1}
cspec = "(G_[0.0, {}] (A))".format(T)
bounds = [-1e-0, 1e-0]

error_bounds = [[mdiff.eps, None], [mdiff.eta, None], [mdiff.nu, None]]
error_bounds = None

cs = casestudy.build_cs(
    sosys, [d0, v0], g, cregions, cspec,
    discretize_system=False, bounds=bounds, error_bounds=None,
    pset=None, f=None)
