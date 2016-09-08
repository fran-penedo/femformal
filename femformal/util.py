import networkx as nx
import matplotlib.pyplot as plt

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
    return next(((i, x[1] - x[0]) for i, x in enumerate(zip(a, b)) if x[0] != x[1]))


def subst_spec_labels(spec, regions):
    res = spec
    for k, v in regions.items():
        res = res.replace(k, state_label(v))
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

def draw_ts(ts):
    nx.draw_networkx(ts)
    plt.show()
