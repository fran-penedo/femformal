import numpy as np
from matplotlib import pyplot as plt

from femformal.core.fem import mech2d
from femformal.core import system as sys

length = 16.0
width = 2.0
mults = [1, 2, 4]
elem_num_x = [4 * mult for mult in mults]
elem_num_y = [2 * mult for mult in mults]
xss = [np.linspace(0, length, e + 1) for e in elem_num_x]
yss = [np.linspace(0, width, e + 1) for e in elem_num_y]
C = np.array([[1.346153846153846e+07, 5.769230769230769e+06, 0.000000000000000e+00],
                       [5.769230769230769e+06, 1.346153846153846e+07, 0.000000000000000e+00],
                       [0.000000000000000e+00, 0.000000000000000e+00, 3.846153846153846e+06]])
E = 1e7
rho = 8e3
traction_templ = mech2d.parabolic_traction(length, width)
P = -50.0
traction = lambda x, y: traction_templ(x, y, P)
force = None
# force = np.array([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.796875000000000e-01, -3.000000000000000e+00, 2.656250000000000e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.656250000000000e-01, 0.000000000000000e+00, 5.468750000000001e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.468750000000001e-02])
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


dts = [.80 * (width / e) / np.sqrt(E / rho) for e in elem_num_y]
T = 50.0
sosys_list = [mech2d.mech2d(xs, ys, rho, C, g, force, dt, traction)
              for xs, ys, dt in zip(xss, yss, dts)]
d0s, v0s = zip(*[mech2d.state(u0, du0, sosys.mesh.nodes_coords, g) for sosys in sosys_list])

trajectories = [sys.newm_integrate(sosys, d0, v0, T, sosys.dt)
                for sosys, d0, v0 in zip(sosys_list, d0s, v0s)]

fig, axes = plt.subplots(1,1)
for i in range(len(dts)):
    axes.plot(np.arange(0, T + dts[i] / 2.0, dts[i]), trajectories[i][0][:, elem_num_x[i] * 2 + 1])

axes.autoscale()
plt.show()
