import numpy as np

from femformal.core import logic, casestudy, system as sys
from femformal.core.fem import mech2d
from examples.mech2d.results import m2d_maxdiff_results as mdiff


length = 16
width = 2
elem_num_x = 4
elem_num_y = 2
xs = np.linspace(0, length, elem_num_x + 1)
ys = np.linspace(0, width, elem_num_y + 1)
C = np.array([[1.346153846153846e+07, 5.769230769230769e+06, 0.000000000000000e+00],
                       [5.769230769230769e+06, 1.346153846153846e+07, 0.000000000000000e+00],
                       [0.000000000000000e+00, 0.000000000000000e+00, 3.846153846153846e+06]])
rho = 8e3
force = None
u0 = lambda x, y: [0.0, 0.0]
du0 = lambda x, y: [0.0, 0.0]

def g(x, y):
    if np.isclose(x, 0.0) and np.isclose(y, 0.0):
        return [0.0, 0.0]
    elif np.isclose(x, 0.0) and np.isclose(y, width):
        return [0.0, None]
    elif np.isclose(y, 0.0):
        return [0.0, None]
    else:
        return [None, None]


dt = 0.01
T = 5.0

traction_templ = mech2d.parabolic_traction(length, width)
input_dt = 1.0
pwlf = sys.PWLFunction(np.linspace(0, T, round(T / input_dt) + 1), ybounds=[-5e3, 0.0], x=None)
fset = pwlf.pset()
d_par = 0.0
v_par = 0.0
dset = np.array([[1, d_par], [-1, d_par]])
vset = np.array([[1, v_par], [-1, v_par]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]

sosys = mech2d.mech2d(xs, ys, rho, C, g, force, dt, None)
d0, v0 = mech2d.state(u0, du0, sosys.mesh.nodes_coords, g)

class FF:
    def __init__(self, pwlf):
        self.pwlf = pwlf
        self.memoize = {}

    @property
    def ys(self):
        return pwlf.ys

    def __call__(self, t, parameters, node):
        pwlf.ys = parameters
        if t in self.memoize:
            traction_force = self.memoize[t]
        else:
            traction = lambda x, y: traction_templ(x, y, pwlf(t))
            traction_force = mech2d.traction_nodal_force(traction, sosys.mesh)
            self.memoize[t] = traction_force
        return traction_force[node]


ff = FF(pwlf)


# apc1 = logic.APCont2D(1, np.array([[16, 1], [16, 2]]), '<', lambda x, y: -30e-6, lambda x, y: 0.0)
apc1 = logic.APCont2D(1, np.array([[16, 1.0], [16, 2]]), '<', lambda x, y: 0.0, lambda x, y: 0.0)
cregions = {'A': apc1}
cspec = "(G_[4.0, 5.0] (A))"
bounds = [-1e-0, 1e-0]
mdiff.eps = mdiff.eps[:15]
mdiff.eta = mdiff.eta[:8]
mdiff.nu = mdiff.nu[:30]
error_bounds = [(mdiff.eps, None), (mdiff.eta, None), (mdiff.nu, None)]

cs = casestudy.build_cs(
    sosys, [d0, v0], g, cregions, cspec,
    discretize_system=False, bounds=bounds, error_bounds=error_bounds,
    pset=[dset, vset, fset], f=[fd, fv, ff])
