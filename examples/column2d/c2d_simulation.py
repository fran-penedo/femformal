import numpy as np

from femformal.core import system as sys, draw_util as draw, logic, casestudy
from femformal.core.fem import mech2d as mech2d


length = 16
width = 2
mult = 2
elem_num_x = 4 * mult
elem_num_y = 2 * mult
xs = np.linspace(0, length, elem_num_x + 1)
ys = np.linspace(0, width, elem_num_y + 1)
C = np.array([[1.346153846153846e+07, 5.769230769230769e+06, 0.000000000000000e+00],
                       [5.769230769230769e+06, 1.346153846153846e+07, 0.000000000000000e+00],
                       [0.000000000000000e+00, 0.000000000000000e+00, 3.846153846153846e+06]])
rho = 8e3
P = -10000
U = 0.90
def traction_templ(x, y, U):
    if np.isclose(x, length):
        y_m = U * width
        if y < U:
            ret = [-5000 + y * (P / y_m), 0.0]
        else:
            ret = [-5000 + P - (y - y_m) / (P / (width - y_m)), 0.0]
    else:
        ret = [0.0, 0.0]

    return np.array(ret)

traction = lambda x, y: traction_templ(x, y, U)
force = None
u0 = lambda x, y: [0.0, 0.0]
du0 = lambda x, y: [0.0, 0.0]

def g(x, y):
    if np.isclose(x, 0.0):
        return [0.0, 0.0]
    else:
        return [None, None]


dt = 0.01
T = 50.0
sosys = mech2d.mech2d(xs, ys, rho, C, g, force, dt, traction)

d0, v0 = mech2d.state(u0, du0, sosys.mesh.nodes_coords, g)

v = .1
apc1 = logic.APCont2D(1, np.array([[0.0, 0], [16, 0]]), '<',
                      lambda x, y: v * (-x * x / 64.0 + x / 4.0),
                      lambda x, y: v * (-x / 32.0 + 1/4.0))
cregions = {'A': apc1}
cspec = "(G_[4.0, 5.0] (A))"
bounds = [-1e-0, 1e-0]

cs = casestudy.build_cs(
    sosys, [d0, v0], g, cregions, cspec,
    discretize_system=False, bounds=bounds, error_bounds=None,
    pset=None, f=None)
