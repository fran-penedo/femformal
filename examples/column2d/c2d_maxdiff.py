import numpy as np

from femformal.core import casestudy, system as sys
from femformal.core.fem import mech2d as mech2d

N = 2

length = 16.0
width = 2.0
C = np.array([[1.346153846153846e+07, 5.769230769230769e+06, 0.000000000000000e+00],
                       [5.769230769230769e+06, 1.346153846153846e+07, 0.000000000000000e+00],
                       [0.000000000000000e+00, 0.000000000000000e+00, 3.846153846153846e+06]])
rho = 8e3
E = 1.e7
mult = 2
elem_num_x = 4 * mult
elem_num_y = 2 * mult
xs = np.linspace(0, length, elem_num_x + 1)
ys = np.linspace(0, width, elem_num_y + 1)
u0 = lambda x, y: [0.0, 0.0]
du0 = lambda x, y: [0.0, 0.0]
force = None

def g(x, y):
    if np.isclose(x, 0.0):
        return [0.0, 0.0]
    else:
        return [None, None]


dt = .80 * (width / elem_num_y) / np.sqrt(E / rho)
sosys = sys.ControlSOSystem.from_sosys(mech2d.mech2d(xs, ys, rho, C, g, force, dt), None)

mult_t = 8
elem_num_x_t = elem_num_x * mult_t
elem_num_y_t = elem_num_y * mult_t
xs_t = np.linspace(0, length, elem_num_x_t + 1)
ys_t = np.linspace(0, width, elem_num_y_t + 1)
force_t = None

dt_t = .80 * (width / elem_num_y_t) / np.sqrt(E / rho)
sosys_t = sys.ControlSOSystem.from_sosys(mech2d.mech2d(xs_t, ys_t, rho, C, g, force_t, dt_t), None)

P = -10000
def traction_templ(x, y, U):
    if np.isclose(x, length):
        y_m = U * width
        if y < y_m:
            ret = [-5000 + y * (P / y_m), 0.0]
        else:
            ret = [-5000 + P - (y - y_m) / (P / (width - y_m)), 0.0]
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
input_dt = 1.0

tlims = [int(round(0.0 / dt)) * dt, (int(round(T / dt)) - 1) * dt]
xlims = np.array([[0.0, 0.0], [length, width]])
fbounds = {'xs': np.linspace(0, T, round(T / input_dt) + 1),
           'ybounds': [0.0, 1.0]}

mds = casestudy.max_diff(
    sosys, g, tlims, xlims, sosys_t,
    ([u0, du0], fbounds), [ic_id_sample, f_sample],
    n=N, pw=True
)

mdxs, mdtxs = casestudy.max_der_diff(
    sosys, g, tlims,
    ([u0, du0], fbounds), [ic_id_sample, f_sample],
    n=N
)

print "eps = {}".format(mds.__repr__())
print "eta = {}".format(mdxs.__repr__())
print "nu = {}".format(mdtxs.__repr__())
