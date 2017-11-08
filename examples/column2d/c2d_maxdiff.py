import numpy as np

from femformal.core import casestudy, system as sys
from femformal.core.fem import mech2d as mech2d
from examples.column2d.c2d_model import *

n_its = 50

mult_t = 4
elem_num_x_t = elem_num_x * mult_t
elem_num_y_t = elem_num_y * mult_t
xs_t = np.linspace(0, length, elem_num_x_t + 1)
ys_t = np.linspace(0, width, elem_num_y_t + 1)
force_t = None
dt_t = dt / 10

sosys = sys.ControlSOSystem.from_sosys(mech2d.mech2d(xs, ys, rho, C, g, force, dt, q4=False), None)
sosys_t = sys.ControlSOSystem.from_sosys(mech2d.mech2d(xs_t, ys_t, rho, C, g, force_t, dt_t, q4=False), None)

center = 0.45
left = .25
right = .75
def traction_templ(x, y, U):
    y_r = right * width
    y_l = left * width
    if np.isclose(x, length) and y > y_l and y < y_r:
        y_m = center * width
        if y < y_m:
            ret = [(y - y_l) * (U / (y_m - y_l)), 0.0]
        else:
            ret = [U - (y - y_m) * U / (y_r - y_m), 0.0]
    else:
        ret = [0.0, 0.0]

    return np.array(ret)


def f_sample(bounds, g, sys_x, sys_y=None):
    xs = bounds['xs']
    ybounds = bounds['ybounds']
    ys = [np.random.rand() * (ybounds[1] - ybounds[0]) + ybounds[0] for x in xs]
    pwlf = sys.PWLFunction(xs, ys)
    traction_force_x = mech2d.TimeVaryingTractionForce(pwlf, traction_templ, sys_x.mesh)
    if sys_y is not None:
        traction_force_y = mech2d.TimeVaryingTractionForce(pwlf, traction_templ, sys_y.mesh)
        return traction_force_x.traction_force, traction_force_y.traction_force
    else:
        return traction_force_x.traction_force


def ic_id_sample(bounds, g, sys_x, sys_y=None):
    u0, du0 = bounds
    d0, v0 = mech2d.state(u0, du0, sys_x.mesh.nodes_coords, g)

    if sys_y is not None:
        d0_y, v0_y = mech2d.state(u0, du0, sys_y.mesh.nodes_coords, g)
        return [d0, v0], [d0_y, v0_y]
    else:
        return [d0, v0]


T = 5.0
input_dt = .75

tlims = [int(round(0.0 / dt)) * dt, (int(round(T / dt)) - 1) * dt]
xlims = np.array([[0.0, 0.0], [length, width]])
fbounds = {'xs': np.linspace(0, T, round(T / input_dt) + 1),
           'ybounds': [-4e3, 0.0]}

mds = casestudy.max_diff(
    sosys, g, tlims, xlims, sosys_t,
    ([u0, du0], fbounds), [ic_id_sample, f_sample],
    n=n_its, pw=True
)

mdxs, mdtxs = casestudy.max_der_diff(
    sosys, g, tlims,
    ([u0, du0], fbounds), [ic_id_sample, f_sample],
    n=n_its, compute_derivative=True
)

print "eps = {}".format(mds.__repr__())
print "eta = {}".format(mdxs.__repr__())
print "nu = {}".format(mdtxs.__repr__())
