import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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
def ap_cont_to_disc(apcont, xpart):
    r = apcont.r
    i_min = bisect_left(xpart, apcont.A[0])
    i_max = bisect_left(xpart, apcont.A[1])
    m = {i : apcont.p(xpart[i]) for i in range(i_min, i_max + 1)}
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



_figcounter = 0

def draw_ts(ts, prefix=None):
    global _figcounter

    nx.draw_networkx(ts)

    if prefix is not None:
        plt.savefig(prefix + str(_figcounter) + '.svg')
        plt.show()
        _figcounter += 1
    else:
        plt.show()


def draw_ts_2D(ts, partition, prefix=None):
    global _figcounter

    if len(label_state(ts.nodes()[0])) != 2:
        raise ValueError("Expected TS from 2D partition")
    if len(partition) != 2:
        raise ValueError("Expected 2D partition")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _draw_grid(partition, ax)
    for e in ts.edges():
        _draw_edge(e, partition, ax)

    if prefix is not None:
        fig.savefig(prefix + str(_figcounter) + '.svg')
        plt.close(fig)
        plt.show()
        _figcounter += 1
    else:
        plt.show()


def _draw_edge(e, partition, ax):
    f = label_state(e[0])
    t = label_state(e[1])
    i, d = first_change(f, t)

    fcenter, fsize = _box_dims(partition, f)
    tcenter, tsize = _box_dims(partition, t)

    if d != 0:
        fcenter += d * fsize / 4.0
        tcenter += d * fsize / 4.0
        tcenter[i] -= 2 * d * fsize[i] / 4.0
        ax.arrow(*np.hstack([fcenter, (tcenter - fcenter)]), color='b', width=.01,
                 head_width=.1, head_starts_at_zero=False)
    else:
        ax.plot(*fcenter, color='b', marker='x', markersize=10)

def _box_dims(partition, state):
    bounds = np.array([[partition[i][s], partition[i][s+1]]
                       for i, s in enumerate(state)])
    return np.mean(bounds, axis=1), bounds[:,1] - bounds[:,0]

def _draw_grid(partition, ax):
    limits = np.array([[p[0], p[-1]] for p in partition])
    for i, p in enumerate(partition):
        data = limits.copy()
        for x in p:
            data[i][0] = data[i][1] = x
            ax.plot(data[0], data[1], color='k', linestyle='--', linewidth=2)




