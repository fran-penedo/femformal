import numpy as np
from .util import project_list, project_regions, label_state, list_extr_points
import util
from .ts import abstract, state_n
from .modelcheck import check_spec

import logging
logger = logging.getLogger('FEMFORMAL')


def verify(system, partition, regions, init_states, spec, depth, **kwargs):
    dims = list(range(system.n))
    while len(dims) > 0:
        d = dims.pop(0)
        m = modelcheck(system, d, partition, regions, init_states, spec, depth)
        if m.sat:
            dims = [i for i in dims if i not in m.dchecked]
        else:
            return False
    return True


def modelcheck(system, dim, partition, regions, init_states, spec, depth):
    indices = [dim]
    subs = system.subsystem(indices)
    p_partition = project_list(partition, indices)
    p_init_states = [project_list(state, indices) for state in init_states]
    ts = abstract(subs, p_partition,
                  list_extr_points(project_list(
                      partition, system.pert_indices(indices))))
    logger.debug(ts.adj)
    # util.draw_ts(ts)
    init = [state_n(ts, state) for state in p_init_states]
    d = 1
    while d <= depth and d < system.n:
        sat, p = check_spec(ts, spec, project_regions(regions, indices), init)
        if sat:
            return ModelcheckResult(True, [])
        else:
            # l = random.choice(p)
            # s = label_state(l)
            # split_state()
            # d = len(s) - 1
            indices += system.pert_indices(indices)
            logger.debug(indices)
            subs = system.subsystem(indices)
            p_partition = project_list(partition, indices)
            p_init_states = [project_list(state, indices) for state in init_states]
            ts = abstract(subs, p_partition,
                        list_extr_points(project_list(
                            partition, system.pert_indices(indices))))
            util.draw_ts(ts)
            init = [state_n(ts, state) for state in p_init_states]
            d = len(indices)

    return ModelcheckResult(False, [])


def split_state(ts, state, system, partition, indices, ext_indices):
    label = state_label(state)
    new_indices = indices + ext_indices
    ppartition = project_list(partition, new_indices)
    subsystem = system.subsystem(new_indices)
    pert_bounds = project_list(partition, subsystem.pert_indices)[[0, -1]]
    new_states = [indices + list(x) for x in it.product(*[
        list(range(len(partition[i]))) for i in ext_indices])]

    new_nodes = [(state_label(s),
              {'rect': np.array([[p[i], p[i+1]] for p, i in zip(ppartition, s)]),
               'state': s,
               'system': subsystem})
             for s in new_states]
    ts.add_nodes_from(new_nodes)

    for succ in ts.successors(label):
        ts.add_transition(label, succ, pert_bounds)
    for pred in ts.predecessors(label):
        ts.add_transition(pred, label, pert_bounds)
    for x, y in it.product(*new_states):
        ts.add_transition(state_label(x), state_label(y), pert_bounds)



class ModelcheckResult(object):

    def __init__(self, sat, dchecked):
        self.sat = sat
        self.dchecked = dchecked

