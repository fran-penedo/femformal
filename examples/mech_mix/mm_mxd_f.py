from core.fem.mechlinfem import mechlinfem
import core.fem.fem_util as fem
import core.system as s
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

sosys_list = [s.ControlSOSystem.from_sosys(
    mechlinfem(xpart, rho, E, g, f_nodal, dt), None)
    for xpart, f_nodal, dt in zip(xparts, fs_nodal, dts)]

Ntrue = 200
xparttrue = np.linspace(0, L, Ntrue + 1)
f_nodal_true = np.zeros(Ntrue + 1)
# f_nodal_true[-1] = 1000.0
dt_true = min((L / Ntrue) / np.sqrt(E_steel / rho_steel),
           (L / Ntrue) / np.sqrt(E_steel / rho_steel))
systrue = s.ControlSOSystem.from_sosys(
    mechlinfem(xparttrue, rho, E, g, f_nodal_true, dt_true), None)

def f_unif_sample(bounds, g, xpart_x, xpart_y=None):
    xs = bounds['xs']
    ybounds = bounds['ybounds']
    ys = [np.random.rand() * (ybounds[1] - ybounds[0]) + ybounds[0] for x in xs]
    pwlf = s.PWLFunction(xs, ys, x=0)
    def f_x(t):
        f = np.zeros(len(xpart_x))
        f[-1] = pwlf(0, t)
        return f
    if xpart_y is not None:
        def f_y(t):
            f = np.zeros(len(xpart_y))
            f[-1] = pwlf(0, t)
            return f
        return f_x, f_y
    else:
        return f_x


dt = dts[-1]
tlims = [int(round(0.0 / dt)) * dt, int(round(0.3 / dt)) * dt]
xlims = [0, 100000]
bounds = {'xs' : np.linspace(0, 0.3, 7),
          'ybounds' : [0, 5e3]}
mds = [fem.max_diff(system, g, tlims, xlims, systrue,
                    ([u0, v0], bounds),
                     [fem.id_sample, f_unif_sample], n=500, pw=True)
       for system in sosys_list]

mdxs, mdtxs = zip(*[fem.max_der_diff(system, g, tlims,
                    ([u0, v0], bounds),
                     [fem.id_sample, f_unif_sample], n=500)
       for system in sosys_list])

print mds
print mdxs
print mdtxs
