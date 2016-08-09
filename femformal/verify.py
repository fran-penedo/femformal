import numpy as np

import logging
logger = logging.getLogger('FEMFORMAL')


def verify(system, partition, regions, spec, depth, **kwargs):
    dims = list(range(system.n))
    while len(dims) > 0:
        d = dims.pop(0)
        m = modelcheck(system, d, partition, regions, spec, depth)
        if m.sat:
            dims = [i for i in dims if i not in m.dchecked]
        else:
            return False


def modelcheck(system, dim, partition, regions, spec, depth):
    indices = [dim]
    subs = system.subsystem(indices)
    p_partition = project_list(partition, indices)
    ts = abstract(subs, p_partition,
                  project_list(partition, system.pert_indices(indices)))
    d = 0
    while d <= depth:
        sat, p = ts.modelcheck(project_regions(regions, indices), spec)
        if sat:
            return ModelcheckResult(True, None)
        else:
            l = random.choice(p)
            s = label_state(l)
            split_state()
            d = len(s) - 1

    return ModelcheckResult(False, None)


def project_list(l, indices):
    return [l[i] for i in indices]

def project_regions(regions, indices):
    ret = {}
    for key, value in regions.items():
        ret[key] = project_list(value, indices)
    return ret

class ModelcheckResult(object):

    def __init__(self, sat, dchecked):
        self.sat = sat
        self.dchecked = dchecked

