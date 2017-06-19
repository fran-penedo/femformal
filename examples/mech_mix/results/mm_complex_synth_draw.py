import femformal.core.fem.mechlinfem as mechlinfem
import femformal.core.system as sys
import femformal.core.logic as logic
import numpy as np

N = 20
L = 100000
rho_steel = 8e-6
rho_brass = 8.5e-6
E_steel = 200e6
E_brass = 100e6
rho = lambda x: rho_steel if x < 30000 or x > 60000 else rho_brass
E = lambda x: E_steel if x < 30000 or x > 60000 else E_brass
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_steel / rho_steel))
u0 = lambda x: 0.0
du0 = lambda x: 0.0

apc1 = logic.APCont([30000, 60000], ">", lambda x: 4 * x / 100000.0 - 2.0 , lambda x: 4 / 100000.0)
apc2 = logic.APCont([60000, 90000], ">", lambda x: 2 * x / 100000.0 - 0.5 , lambda x: 2 / 100000.0)
apc3 = logic.APCont([60000, 90000], "<", lambda x: 0 * x / 100000.0 - .0 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
# cspec = "G_[0.001, 0.005] (F_[0.0, 0.002] (A) & F_[0.0, 0.002] (B))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
d0, v0 = mechlinfem.state(u0, du0, xpart, g)
pwlf = sys.PWLFunction(np.linspace(0, 0.55, 55/5 + 1),
                       [-337.6068306932366, 2280.3608476576946, 5000.0, 5000.0, 4998.693984662118, 5000.0, 4980.280793930004, 533.1664254838698, 226.35746361104665, 5000.0, 3283.3900480158745, 4708.55867091666])

# [-958.4030047608805, -563.9556819501967, 3280.271225863673, 4986.680454677409, 4849.318890201641, 5000.0, 2538.4206567995843, -1637.3124847021381, 2125.5669024622453, 4790.865644850633, 4639.940881741323, 4568.949393649726]



def f_nodal_control(t):
    f = np.zeros(N + 1)
    f[-1] = pwlf(t)
    return f


csosys = sys.ControlSOSystem.from_sosys(sosys, f_nodal_control)

import matplotlib.pyplot as plt
(fig, ) = sys.draw_sosys(csosys, d0, v0, g, 0.55, animate=False, allonly=False, hold=True)
fig.set_size_inches(3,2)
ax = plt.gcf().get_axes()[0]
labels = ['A', 'B', 'C']
for (key, apc), label in zip(sorted(cregions.items()), labels):
    ax.plot(apc.A, [apc.p(x) for x in apc.A], lw=1, label=label)
# ax.autoscale()
ax.legend(loc='lower left', fontsize='6', labelspacing=0.05, handletextpad=0.1)
ax.set_xticklabels([x / 1000 for x in ax.get_xticks()])
plt.show()
# fig.savefig('fig2.png')
plt.close(fig)

sys.draw_pwlf(pwlf)
fig = plt.gcf()
fig.set_size_inches(3,2)
fig.savefig('synth_input.png')
# plt.show()
plt.close(fig)


ts = [0.0, 0.1, 0.3, 0.37, 0.45, 0.5]
figs = sys.draw_sosys_snapshots(csosys, d0, v0, g, ts, hold=True, ylims=[-1.5, 3.5])

for fig, t in zip(figs, ts):
    fig.set_size_inches(3,2)
    fig.canvas.set_window_title('i3_7')
    ax = fig.get_axes()[0]

    labels = ['A', 'B', 'C']
    for (key, apc), label in zip(sorted(cregions.items()), labels):
        ax.plot(apc.A, [apc.p(x) for x in apc.A], lw=1, label=label)
    # ax.autoscale()
    ax.legend(loc='lower left', fontsize='6', labelspacing=0.05, handletextpad=0.1)
    ax.set_xticklabels([x / 1000 for x in ax.get_xticks()])
    fig.savefig('synth_snaps_t{}.png'.format(str(t).replace('.','_')))

# plt.show()
for fig in figs: plt.close(fig)
