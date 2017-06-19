import femformal.core.fem.mechlinfem as mechlinfem
import femformal.core.fem.fem_util as fem
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
# f_nodal[-1] = 2e6
dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_brass / rho_brass))
fdt_mult = 2
bounds = [-10, 10]
u0 = lambda x: 0.0
du0 = lambda x: 0.0

# apc1 = logic.APCont([30000, 60000], ">", lambda x: 4 * x / 100000.0 - 2.4 , lambda x: 4 / 100000.0)
# apc2 = logic.APCont([60000, 90000], ">", lambda x: 2 * x / 100000.0 - 0.9 , lambda x: 2 / 100000.0)
# apc3 = logic.APCont([60000, 90000], "<", lambda x: 0 * x / 100000.0 - .0 , lambda x: 0.0)
apc1 = logic.APCont([30000, 60000], ">", lambda x: 4 * x / 100000.0 - 1.5 , lambda x: 4 / 100000.0)
apc2 = logic.APCont([60000, 90000], ">", lambda x: 2 * x / 100000.0 - 0 , lambda x: 2 / 100000.0)
apc3 = logic.APCont([60000, 90000], "<", lambda x: 0 * x / 100000.0 - .5 , lambda x: 0.0)
cregions = {'A': apc1, 'B': apc2, 'C': apc3}

cspec = "((G_[0.1, 0.3] ((A) & (B))) & (F_[0.3, 0.4] (C)) & ((G_[0.45, 0.5] (C) | G_[0.45, 0.5] (B))) & (F_[0.5, 0.52] (B)))"
# cspec = "(G_[0.1, 0.3] ((A) & (B)))"
# cspec = "((G_[0.1, 0.3] ((A) & (B))) & (F_[0.3, 0.4] (C)))"

eps = [0.06406569900430464, 0.08734642833508732, 0.09281706976957327, 0.09327501419326811, 0.08742770440826175, 0.06870630767948627, 0.07563917401423514, 0.11897234455073091, 0.12591470276738148, 0.10264731859022369, 0.11414080748384448, 0.09423679595394696, 0.09988201861757917, 0.09569565956479598, 0.10913387529809615, 0.10020372052758661, 0.09437233975604686, 0.08315362318631614, 0.10066159188823764, 0.10874752910880936]
eps_xderiv = [2.98921636961007e-05, 1.7991299025380423e-05, 2.1585677535109455e-05, 2.1585677535109455e-05, 1.976805239478643e-05, 3.521045126322881e-05, 3.8256818323584115e-05, 4.093638513872201e-05, 3.895290703564729e-05, 3.88906749107096e-05, 4.104917973982763e-05, 3.677391327445042e-05, 2.040891005526628e-05, 2.285400412842762e-05, 2.233265260723645e-05, 2.2264446143837445e-05, 2.4058121172539605e-05, 2.57421459814406e-05, 2.6532997491059264e-05]
eta = [0.6928021741926784, 0.6810690474658312, 0.6713795870064214, 0.6636960582007778, 0.6629067623123555, 0.6511736355855082, 1.2826191291691122, 1.2289026041703321, 1.1720444181928809, 1.0794849005805283, 1.0026394955300724, 0.9161983384927268, 0.42115273317245716, 0.39403952707070466, 0.34926713579575797, 0.3103628562987346, 0.2851178185912513, 0.2483825253827634, 0.22327870266328986, 0.1929540982026401]
nu = [0.0, 0.0851698220414267, 0.12830503995667128, 0.17462909669393453, 0.22462629891145, 0.2568208482679676, 0.29578914439163606, 0.3845541848540863, 0.4640705382033672, 0.541587098023897, 0.6215814634522946, 0.7101847880934137, 0.7697230299107128, 0.8011196861388797, 0.8172216993802008, 0.8333761787566916, 0.8758079947818023, 0.8839652503829907, 0.9157223147792903, 0.9224602024872937, 0.9320074403392915]
nu_xderiv = [1.7519107750552705e-05, 1.1080295293846357e-05, 1.3729685498047381e-05, 1.2874081386085235e-05, 1.1080201109513956e-05, 1.2865040195532097e-05, 2.0712914438107414e-05, 2.16597410305003e-05, 2.2497682900320834e-05, 2.0486791572497054e-05, 2.446884069869091e-05, 2.5491861288432437e-05, 2.461536270977701e-05, 2.461536270977701e-05, 2.461536270977701e-05, 2.461536270977701e-05, 2.461536270977701e-05, 2.461536270977701e-05, 2.461536270977701e-05, 2.8337046276549138e-05]

dset = np.array([[1, 0], [-1, 0]])
vset = np.array([[1, 0], [-1, 0]])
fd = lambda x, p: p[0]
fv = lambda x, p: p[0]
pwlf = sys.PWLFunction(np.linspace(0, 0.55, 55/5 + 1), ybounds=[-5e3, 5e3], x=L)
fset = pwlf.pset()


sosys = mechlinfem.mechlinfem(xpart, rho, E, g, f_nodal, dt)
d0, v0 = mechlinfem.state(u0, du0, xpart, g)
cs = fem.build_cs(sosys, None, g, cregions, cspec, discretize_system=False,
                  pset=[dset, vset, fset], f=[fd, fv, pwlf], fdt_mult=fdt_mult,
                  bounds=bounds, eps=eps, eta=eta, nu=nu)
cs.dsystem = cs.system

# print sosys
# print d0
# print v0
print cs.spec
# print dt

# import matplotlib.pyplot as plt
# sys.draw_sosys(sosys, d0, v0, g, 0.3, animate=False, hold=True)
# ax = plt.gcf().get_axes()[0]
# for apc in cregions.values():
#     ax.plot(apc.A, [apc.p(x) for x in apc.A], 'b-', lw=1)
#
# ax.autoscale()
# plt.show()



# Robustness = -0.0778411261527
# Synthesized parameters = [-343.61719204239097, 2403.152383496667, 5000.0, 4995.946099566848, 5000.0, 5000.0, 4964.326287906423, -193.99034702994763, -3152.680129177063, 5000.0, 5000.0, -5000.000000000002]
# Time = 8990.92398596

# Robustness = 1.13098287475
# Synthesized parameters = [-337.6068306951651, 2280.3608476556, 5000.0, 5000.0, 4998.693984664533, 5000.0, 5000.0, 275.76339129718406, -3910.7042516950323, 4851.395958990179, 4953.734028856025, -5000.0]
# Time = 13379.404752

