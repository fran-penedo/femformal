import femformal.core.fem.mech2d as mech2d
import numpy as np
import femformal.core.system as sys
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3
import femformal.core.draw_util as draw

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
force = np.array([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.796875000000000e-01, -3.000000000000000e+00, 2.656250000000000e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.656250000000000e-01, 0.000000000000000e+00, 5.468750000000001e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.468750000000001e-02])
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
T = 50.0
sosys, elems_nodes = mech2d.mech2d(xs, ys, rho, C, g, force, dt)

d0, v0 = mech2d.state(u0, du0, sosys.xpart, g)

d, v = sys.newm_integrate(sosys, d0, v0, T, dt)

ts = np.arange(0, T + 1e-7, dt)

draw.draw_2d_pde_trajectory(d, sosys.xpart, elems_nodes, ts)
