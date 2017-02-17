from fem.mechlinfem import mechlinfem
import fem.fem_util as fem
import femformal.system as s
import numpy as np


Ns = [20]
L = 100.0
rho = .000724
E = 30e6
xparts = [np.linspace(0, L, N + 1) for N in Ns]
g = [0.0, None]
fs_nodal = [np.zeros(N + 1) for N in Ns]
# for f_nodal in fs_nodal:
#     f_nodal[-1] = 1000.0
dt = 5.0 / np.sqrt(E/rho)
u0 = lambda x: 0.0
v0 = lambda x: 0.0

sosys_list = [mechlinfem(xpart, rho, E, g, f_nodal, dt) for xpart, f_nodal in
                zip(xparts, fs_nodal)]

Ntrue = 200
xparttrue = np.linspace(0, L, Ntrue + 1)
f_nodal_true = np.zeros(Ntrue + 1)
# f_nodal_true[-1] = 1000.0
dt_true = (L / Ntrue) / np.sqrt(E / rho)
systrue = mechlinfem(xparttrue, rho, E, g, f_nodal_true, dt_true)

def f_unif_sample(bounds, g, xpart_x, xpart_y=None):
    f_x = np.zeros(len(xpart_x))
    f_x[-1] = np.random.rand() * (bounds[1] - bounds[0]) + bounds[0]

    if xpart_y is not None:
        f_y = np.zeros(len(xpart_y))
        f_y[-1] = f_x[-1]
        return f_x, f_y
    else:
        return f_x

tlims = [int(round(0.001 / dt)) * dt, int(round(0.005 / dt)) * dt]
xlims = [20.0, 80.0]
fbounds = [900.0, 1100.0]
mds = [fem.max_diff(system, g, tlims, xlims, systrue,
                    ([u0, v0], fbounds),
                     [fem.id_sample, f_unif_sample])
       for system in sosys_list]

mdxs, mdtxs = zip(*[fem.max_der_diff(system, g, tlims,
                    ([u0, v0], fbounds),
                     [fem.id_sample, f_unif_sample])
       for system in sosys_list])

print mds
print mdxs
print mdtxs
