from __future__ import division, absolute_import, print_function

import numpy as np

from femformal.core import system as sys

N = 5
L = 1000
rho = 9e-6
E_nominal = 1e5
# sigma_yield = 350e3
# yield_point = sigma_yield / E_nominal
yield_point = .005
hardening_point = .01

E = sys.HybridParameter([
    lambda p: (np.array([-1, 1]) / np.diff(p), yield_point),
    # lambda p: (np.array([[1, -1], [-1, 1]]) / np.diff(p),
    #            np.array([hardening_point, -yield_point])),
    lambda p: (np.array([1, -1]) / np.diff(p), -yield_point)],
    [E_nominal, E_nominal / 2.0])
bigN_deltas = .1
bigN_int_force = 10 * E_nominal * yield_point

# E.p = [0, 50]
# print(E.invariant_representatives())

xpart = np.linspace(0, L, N + 1)

f_nodal = np.zeros(N + 1)
u0 = lambda x: 0.0
du0 = lambda x: 0.0

g = [0.0, None]

dt = .005
T = 0.5

N_t = 200
dt_t = 0.005
f_nodal_t = np.zeros(N_t + 1)
xpart_t = np.linspace(0, L, N_t + 1)
