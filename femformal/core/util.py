import numpy as np

import logging
logger = logging.getLogger(__name__)

def label(name, i, j):
    return name + "_" + str(i) + "_" + str(j)

def unlabel(label):
    sp = label.split("_")
    return sp[0], int(sp[1]), int(sp[2])


def state_label(l):
    return 's' + '_'.join([str(x) for x in l])

def label_state(l):
    return [int(x) for x in l[1:].split('_')]

def long_first(a, b):
    if len(a) > b:
        return a, b
    else:
        return b, a

def first_change(a, b):
    try:
        return next(((i, x[1] - x[0]) for i, x in enumerate(zip(a, b)) if x[0] != x[1]))
    except StopIteration:
        return 0, 0


def make_groups(l, n):
    leftover = len(l) % n
    at = leftover * (n + 1)
    if leftover > 0:
        bigger = make_groups(l[:at], n+1)
    else:
        bigger = []
    rest = l[at:]
    return bigger + [rest[n * i: n * (i + 1)] for i in range(len(rest) / n)]


def project_states(states):
    x = np.array(states)
    return [list(set(c)) for c in x.T]


def project_list(l, indices):
    return [l[i] for i in indices]

def list_extr_points(l):
    return [[x[0], x[-1]] for x in l]

def project_regions(regions, indices):
    ret = {}
    for key, value in regions.items():
        ret[key] = project_list(value, indices)
    return ret

