from __future__ import division, absolute_import, print_function

import numpy as np


length = 16.0e3
width = 1.0e3
mult = 4
elem_num_x = 4 * mult
elem_num_y = 2 * mult
xs = np.linspace(0, length, elem_num_x + 1)
ys = np.linspace(0, width, elem_num_y + 1)
C = 1e-3 * np.array([[1.346153846153846e+07, 5.769230769230769e+06, 0.000000000000000e+00],
                       [5.769230769230769e+06, 1.346153846153846e+07, 0.000000000000000e+00],
                       [0.000000000000000e+00, 0.000000000000000e+00, 3.846153846153846e+06]])
rho = 8e-6

def create_unif_traction_templ(xs, ys, f):
    x_l, x_r = xs
    y_l, y_r = ys
    if np.isclose(x_l, x_r):
        x_apply = x_l
    else:
        x_apply = None
    if np.isclose(y_l, y_r):
        y_apply = y_l
    else:
        y_apply = None

    def traction_templ(x, y, U):
        if x_apply is not None and np.isclose(x_apply, x) and y > y_l and y < y_r:
            ret = f(x, y, U)
        elif y_apply is not None and np.isclose(y_apply, y) and x > x_l and x < x_r:
            ret = f(x, y, U)
        else:
            ret = [0.0, 0.0]
        return np.array(ret)

    return traction_templ

def combine_tractions(tractions):
    def traction_templ(x, y, U):
        return np.sum([traction(x, y, Ut) for traction, Ut in zip(tractions, U)], axis=0)
    return traction_templ

traction1 = create_unif_traction_templ([16e3,16e3], [0, 0.4e3], lambda x, y, U: [U, 0])
traction2 = create_unif_traction_templ([16e3,16e3], [0.6e3, 1.0e3], lambda x, y, U: [U, 0])
traction_templ = combine_tractions([traction1, traction2])

force = None
# v_pert = width * 0.005
# u0 = lambda x, y: [0.0, v_pert * (-x * x / 64.0 + x / 4.0)]
u0 = lambda x, y: [0.0, 0.0]
du0 = lambda x, y: [0.0, 0.0]

def g(x, y):
    if np.isclose(x, 0.0):
        return [0.0, 0.0]
    else:
        return [None, None]


dt = 0.075
T = 5.0

mult_t = 1
elem_num_x_t = elem_num_x * mult_t
elem_num_y_t = elem_num_y * mult_t
xs_t = np.linspace(0, length, elem_num_x_t + 1)
ys_t = np.linspace(0, width, elem_num_y_t + 1)
force_t = None
dt_t = dt
