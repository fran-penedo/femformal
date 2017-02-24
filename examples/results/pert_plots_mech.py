import numpy as np
import matplotlib.pyplot as plt
import matplotlib

eps = 9.6884306133304676e-16

eta = [ 0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452,
        0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452,
        0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452,
        0.00036452,  0.00036452,  0.00036452,  0.00036452,  0.00036452]
nu = [ 0.        ,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226,  0.00018226,  0.00018226,  0.00018226,  0.00018226,
        0.00018226]

matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'figure.autolayout': True})

# for i in [1, 4]:
L = 100
fig, ax = plt.subplots()
fig.set_size_inches(3,2)
ax.set_xlabel('x')
ax.set_ylim((0, 5e-4))
x = np.linspace(0, L, 21)
line1 = ax.hlines(np.array(eta) / 2  , x[:-1], x[1:], 'r', label='$\eta h / 2$')
line2 = ax.plot(x, nu, 'go', markersize=3, label='$\\nu \Delta t$')
line3 = ax.hlines(eps, 0, L, 'b', label='$\delta$')
ax.legend(loc='upper center')
# plt.show()
fig.savefig('pert_plot_mech.png')


