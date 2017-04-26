from femformal.core.fem.mechlinfem import mechlinfem
import femformal.core.fem.fem_util as fem
import femformal.core.system as s
import numpy as np


Ns = [20]
L = 100000
rho_steel = 8e-6
rho_brass = 8.5e-6
E_steel = 200e6
E_brass = 100e6
rho = lambda x: rho_steel if x < 30000 or x > 60000 else rho_brass
E = lambda x: E_steel if x < 30000 or x > 60000 else E_brass
xparts = [np.linspace(0, L, N + 1) for N in Ns]
g = [0.0, None]
fs_nodal = [np.zeros(N + 1) for N in Ns]
dts = [min((L / N) / np.sqrt(E_steel / rho_steel),
           (L / N) / np.sqrt(E_steel / rho_steel)) for N in Ns]
u0 = lambda x: 0.0
v0 = lambda x: 0.0

sosys_list = [mechlinfem(xpart, rho, E, g, f_nodal, dt) for xpart, f_nodal, dt in
                zip(xparts, fs_nodal, dts)]

Ntrue = 200
xparttrue = np.linspace(0, L, Ntrue + 1)
f_nodal_true = np.zeros(Ntrue + 1)
# f_nodal_true[-1] = 1000.0
dt_true = min((L / Ntrue) / np.sqrt(E_steel / rho_steel),
           (L / Ntrue) / np.sqrt(E_steel / rho_steel))
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


dt = dts[-1]
tlims = [int(round(0.0 / dt)) * dt, int(round(0.3 / dt)) * dt]
xlims = [[10000.0, 25000.0], [35000, 55000], [65000, 90000]]
fbounds = [1.9e3, 2.1e3]
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
