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
for f_nodal in fs_nodal:
    f_nodal[-1] = 1000.0
dt = 5.0 / np.sqrt(E/rho)

sosys_list = [mechlinfem(xpart, rho, E, g, f_nodal) for xpart, f_nodal in
                zip(xparts, fs_nodal)]

Ntrue = 200
xparttrue = np.linspace(0, L, Ntrue + 1)
f_nodal_true = np.zeros(Ntrue + 1)
f_nodal_true[-1] = 1000.0
systrue = mechlinfem(xparttrue, rho, E, g, f_nodal_true)
dt_true = (L / Ntrue) / np.sqrt(E / rho)

tlims = [0.001, 0.007]
xlims = [20.0, 80.0]
mds = [fem.max_diff(system, dt, xpart, g, tlims, xlims, systrue,
                    dt_true, xparttrue, [0.0, 7e-3], fem.so_lin_sample)
       for system, xpart in zip(sosys_list, xparts)]

# mdxs = [fem.max_xdiff(system, dt, xpart, t0, tt, T)
#         for system, xpart in zip(systems, xparts)]
# mdtxs = [fem.max_tdiff(system, dt, xpart, t0, tt, T)
#         for system, xpart in zip(dsystems, xparts)]

print mds
# print mdxs
# print mdtxs
