import numpy as np
from matplotlib import pyplot as plt

from femformal.core import system as s
from femformal.core.fem.mechlinfem import mechlinfem, state


Ns = [20, 160]
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
           (L / N) / np.sqrt(E_brass / rho_brass)) for N in Ns]
u0 = lambda x: 0.0
uu0 = lambda x: 0.0

sosys_list = [s.ControlSOSystem.from_sosys(
    mechlinfem(xpart, rho, E, g, f_nodal, dt), None)
    for xpart, f_nodal, dt in zip(xparts, fs_nodal, dts)]

pwlf = s.PWLFunction([0, 0.1, 0.2, 0.3, 0.4, 0.5], [5e3, -5e3, 5e3, -5e3, 5e3, -5e3])

def ff(xp):
    def f_x(t):
        f = np.zeros(len(xp))
        f[-1] = pwlf(t)
        return f
    return f_x

ds = []
vs = []
for system in sosys_list:
    system.add_f_nodal(ff(system.xpart))
    d0, v0 = state(u0, uu0, system.xpart, g)

    dx, vx = s.newm_integrate(system, d0, v0, 0.5, system.dt)
    ds.append(dx)
    vs.append(vx)

fig, axes = plt.subplots(2,2)
for i in range(len(Ns)):
    axes[0,0].plot(np.arange(0, 0.5 + dts[i] / 2.0, dts[i]), ds[i][:, Ns[i]/2])
    # axes[0,1].plot(np.arange(0, 0.5 + dts[i] / 2.0, dts[i]), vs[i][:, Ns[i]/2])
    axes[0,1].plot(np.arange(0, 0.5 + dts[i] / 2.0, dts[i]),
                   (np.diff(ds[i]) / (L / Ns[i]))[:, Ns[i]/2])

s.draw_pwlf(pwlf, axes=axes[1,0])
axes[0,0].autoscale()
axes[0,1].autoscale()
plt.show()
