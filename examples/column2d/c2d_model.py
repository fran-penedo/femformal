from __future__ import division, absolute_import, print_function

import numpy as np


length = 16.0e3
width = 1.0e3
mult = 4
elem_num_x = 4 * mult
elem_num_y = 2 * mult
xs = np.linspace(0, length, elem_num_x + 1)
ys = np.linspace(0, width, elem_num_y + 1)
C = 1e-3 * np.array(
    [
        [1.346153846153846e07, 5.769230769230769e06, 0.000000000000000e00],
        [5.769230769230769e06, 1.346153846153846e07, 0.000000000000000e00],
        [0.000000000000000e00, 0.000000000000000e00, 3.846153846153846e06],
    ]
)
rho = 8e-6
center = 0.45
left = 0.25
right = 0.75


def traction_templ(x, y, U):
    # if x == length:
    y_r = right * width
    y_l = left * width
    if np.isclose(x, length) and y > y_l and y < y_r:
        y_m = center * width
        if y < y_m:
            ret = [(y - y_l) * (U / (y_m - y_l)), 0.0]
        else:
            ret = [U - (y - y_m) * U / (y_r - y_m), 0.0]
    else:
        ret = [0.0, 0.0]

    return np.array(ret)


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
