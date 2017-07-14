import matplotlib
import numpy as np
from matplotlib import pyplot as plt


eps = 0.46369130943083958
eta = [ 0.13755909,  0.13126048,  0.13162718,  0.1222216 ,  0.13138857,
        0.84161904,  0.26558981,  0.27607333,  0.2477102 ,  0.25094865,
        0.26652575,  0.26906774,  1.22645221,  0.1283975 ,  0.12622162,
        0.11715973,  0.12185615,  0.11242183,  0.10054052,  0.08407562]
nu = [ 0.        ,  0.03420027,  0.05542994,  0.06004345,  0.06031116,
        0.06031116,  0.13057483,  0.12971402,  0.125502  ,  0.13602502,
        0.1138558 ,  0.10627603,  0.13725949,  0.08015514,  0.0780944 ,
        0.07784882,  0.07825857,  0.07887112,  0.08361303,  0.08610627,
        0.10080335]

matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'figure.autolayout': True})

# for i in [1, 4]:
L = 100
fig, ax = plt.subplots()
fig.set_size_inches(2,2)
ax.set_xlabel('Location x (m)')
ax.set_ylim((0, 1))
x = np.linspace(0, L, 21)
line1 = ax.hlines(np.array(eta) / 2  , x[:-1], x[1:], 'r', label='$\eta h / 2$')
line2 = ax.plot(x, nu, 'go', markersize=3, label='$\\nu \Delta t$')
line3 = ax.hlines(eps, 0, L, 'b', label='$\delta$')
ax.legend(loc='upper left', labelspacing=0.2)
# plt.show()
fig.savefig('pert_plot_mech.png')


