from __future__ import division, absolute_import, print_function

import numpy as np


length = 16.0
width = 1.0
mult = 4
elem_num_x = 4 * mult
elem_num_y = 2 * mult
xs = np.linspace(0, length, elem_num_x + 1)
ys = np.linspace(0, width, elem_num_y + 1)
C = np.array([[1.346153846153846e+07, 5.769230769230769e+06, 0.000000000000000e+00],
                       [5.769230769230769e+06, 1.346153846153846e+07, 0.000000000000000e+00],
                       [0.000000000000000e+00, 0.000000000000000e+00, 3.846153846153846e+06]])
rho = 8e3
center = 0.45
left = .25
right = .75
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


dt = 0.1
T = 3.0
