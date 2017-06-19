import matplotlib.pyplot as plt
import numpy as np

import femformal.core.fem.mechnlfem as mechnlfem
import femformal.core.logic as logic
import femformal.core.system as sys


N = 4
L = 100000
rho_steel = 8e-6
# rho_brass = 8.5e-6
E_steel = 200e6
sigma_yield_steel = 350e3
yield_point_steel = sigma_yield_steel / E_steel
def E_steel_hybrid(x, u):

    du = np.diff(u) / np.diff(x)
    # if du < yield_point_steel or not (x[0] > 50000 and x[0] < 60000):
    if du < yield_point_steel:
        return E_steel
    else:
        return - E_steel / 2.0


# E_brass = 100e6
# rho = lambda x: rho_steel if x < 30000 or x > 60000 else rho_brass
# E = lambda x: E_steel if x < 30000 or x > 60000 else E_brass
rho = rho_steel
E = E_steel_hybrid
xpart = np.linspace(0, L, N + 1)
g = [0.0, None]
f_nodal = np.zeros(N + 1)
dt = 0.5 * min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_steel / 2 * rho_steel))
u0 = lambda x: 0.0
du0 = lambda x: 0.0

apc1 = logic.APCont([30000, 60000], ">", lambda x: 4 * x / 100000.0 - 1.7 , lambda x: 4 / 100000.0)
apc2 = logic.APCont([60000, 90000], ">", lambda x: 2 * x / 100000.0 - 0.3 , lambda x: 2 / 100000.0)
apc3 = logic.APCont([30000, 60000], "<", lambda x: 0 * x / 100000.0 - .5 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

# cspec = "((F_[1, 10] (A)) & (G_[1, 10] (B)))"
# cspec = "(F_[1, 10] (A))"
# cspec = "G_[0.001, 0.005] (F_[0.0, 0.002] (A) & F_[0.0, 0.002] (B))"
# cspec = "F_[{}, {}] (B)".format(54 * dt, 54 * dt + 0.002)
# cspec = "F_[0.001, 0.02] (B)"

sosys = mechnlfem.mechnlfem(xpart, rho, E, g, f_nodal, dt)
d0, v0 = mechnlfem.state(u0, du0, xpart, g)
# pwlf = sys.PWLFunction(np.linspace(0, 0.55, 55/5 + 1),
#                        [-72.65567723137839, 146.7141767525155, 354900.7143976258585, 500000.0, 409900.154336041715, 437300.835495768179, 338900.069760612807, -98300.4393233484677, 174500.9676272778236, 500000.0, 434600.830236004789, 282000.8350764217453])
pwlf = sys.PWLFunction([0, 0.15, 0.15, 1.0], [1.4e5, 1.4e5, 0.0, 0.0])

def f_nodal_control(t):
    f = np.zeros(N + 1)
    f[-1] = pwlf(t)
    return f


csosys = sys.ControlSOSystem.from_sosys(sosys, f_nodal_control)
sys.draw_sosys(csosys, d0, v0, g, 1.0, animate=False, allonly=False, hold=True)
# fig = plt.gcf()
# fig.set_size_inches(3,2)
# ax = plt.gcf().get_axes()[0]
# labels = ['A', 'B', 'C']
# for (key, apc), label in zip(sorted(cregions.items()), labels):
#     print key, label
#     ax.plot(apc.A, [apc.p(x) for x in apc.A], lw=1, label=label)
# ax.autoscale()
# ax.legend(loc='lower left', fontsize='6', labelspacing=0.05, handletextpad=0.1)
# ax.set_xticklabels([x / 1000 for x in ax.get_xticks()])
plt.show()
# fig.savefig('fig2.png')
plt.close(fig)

sys.draw_pwlf(pwlf)
fig = plt.gcf()
fig.set_size_inches(3,2)
# fig.savefig('fig2force.png')
# plt.show()
