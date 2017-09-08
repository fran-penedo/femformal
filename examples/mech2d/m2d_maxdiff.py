import numpy as np

from femformal.core import casestudy
from femformal.core.fem import mech2d


N = 3

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
# force = np.array([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.796875000000000e-01, -3.000000000000000e+00, 2.656250000000000e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.656250000000000e-01, 0.000000000000000e+00, 5.468750000000001e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.468750000000001e-02])
force = np.zeros(2 * len(xs) * len(ys))
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
sosys = mech2d.mech2d(xs, ys, rho, C, g, force, dt)
# d0, v0 = mech2d.state(u0, du0, sosys.mesh.nodes_coords, g)


elem_num_x_t = elem_num_x * 8
elem_num_y_t = elem_num_y * 8
xs_t = np.linspace(0, length, elem_num_x_t + 1)
ys_t = np.linspace(0, width, elem_num_y_t + 1)
force_t = np.zeros(2 * len(xs_t) * len(ys_t))

dt_t = 0.001
sosys_t = mech2d.mech2d(xs_t, ys_t, rho, C, g, force_t, dt_t)

def f_sample(bounds, g, sys_x, sys_y=None):
    P_x = np.random.rand() * (bounds[1] - bounds[0]) + bounds[0]
    traction = mech2d.parabolic_traction(length, width)
    f_x = mech2d.traction_nodal_force(
        lambda x, y: traction(x, y, P_x), sys_x.mesh)
    for n in range(sys_x.mesh.nnodes):
        for i in range(2):
            if g(*sys_x.mesh.nodes_coords[n])[i] is not None:
                f_x[2*n + i] = 0.0
    if sys_y is not None:
        P_y = np.random.rand() * (bounds[1] - bounds[0]) + bounds[0]
        traction = mech2d.parabolic_traction(length, width)
        f_y = mech2d.traction_nodal_force(
            lambda x, y: traction(x, y, P_y), sys_y.mesh)
        for n in range(sys_y.mesh.nnodes):
            for i in range(2):
                if g(*sys_y.mesh.nodes_coords[n])[i] is not None:
                    f_y[2*n + i] = 0.0
        return f_x, f_y
    else:
        return f_x

def ic_id_sample(bounds, g, sys_x, sys_y=None):
    u0, du0 = bounds
    d0, v0 = mech2d.state(u0, du0, sys_x.mesh.nodes_coords, g)

    if sys_y is not None:
        d0_y, v0_y = mech2d.state(u0, du0, sys_y.mesh.nodes_coords, g)
        return [d0, v0], [d0_y, v0_y]
    else:
        return [d0, v0]



tlims = [int(round(0.0 / dt)) * dt, int(round(5.0 / dt)) * dt]
xlims = np.array([[0.0, 0.0], [length, width]])
fbounds = [-1.5, -0.5]

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

print mds
print mdxs
print mdtxs

