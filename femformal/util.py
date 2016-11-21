import numpy as np
import itertools as it
from bisect import bisect_left, bisect_right

import logging
logger = logging.getLogger('FEMFORMAL')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s %(module)s:%(lineno)d:%(funcName)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logger.setLevel(logging.DEBUG)

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


class APCont(object):
    def __init__(self, A, r, p):
        # A : [x_min, x_max] (np.array)
        self.A = A
        # r == 1: f < p, r == -1: f > p
        self.r = r
        self.p = p

class APDisc(object):
    def __init__(self, r, m):
        self.r = r
        # m : i -> p(x_i)
        self.m = m


# xpart : [x_i] (list)
# FIXME TS based approach probably has wrong index assumption
def ap_cont_to_disc(apcont, xpart):
    r = apcont.r
    N1 = len(xpart)
    i_min = max(bisect_left(xpart, apcont.A[0]), 0)
    i_max = min(bisect_left(xpart, apcont.A[1]), N1 - 1)
    m = {i - 1 : apcont.p(xpart[i]) for i in range(i_min, i_max + 1)
         if i > 0 and i < N1 - 1}
    return APDisc(r, m)

def project_apdisc(apdisc, indices, tpart):
    state_indices = []
    for i in indices:
        if i in apdisc.m:
            if apdisc.r == 1:
                bound_index = bisect_right(tpart, apdisc.m[i]) - 1
                state_indices.append(list(range(bound_index)))
            if apdisc.r == -1:
                bound_index = bisect_left(tpart, apdisc.m[i])
                state_indices.append(list(range(bound_index, len(tpart) - 1)))

    return list(it.product(*state_indices))

def subst_spec_labels_disc(spec, regions):
    res = spec
    for k, v in regions.items():
        replaced = "(" + " & ".join(["({0} {1} {2})".format(
            i, "<" if v.r == 1 else ">", p) for (i, p) in v.m.items()]) + ")"
        res = res.replace(k, replaced)
    return res


def subst_spec_labels(spec, regions):
    res = spec
    for k, v in regions.items():
        if any(isinstance(el, list) or isinstance(el, tuple) for el in v):
            replaced = "(" + " | ".join(["(state = {})".format(state_label(s))
                                          for s in v]) + ")"
        else:
            replaced = "(state = {})".format(state_label(v))
        res = res.replace(k, replaced)
    return res


def project_list(l, indices):
    return [l[i] for i in indices]

def list_extr_points(l):
    return [[x[0], x[-1]] for x in l]

def project_regions(regions, indices):
    ret = {}
    for key, value in regions.items():
        ret[key] = project_list(value, indices)
    return ret

def project_apdict(apdict, indices, tpart):
    ret = {}
    for key, value in apdict.items():
        ret[key] = project_apdisc(value, indices, tpart)
    return ret



