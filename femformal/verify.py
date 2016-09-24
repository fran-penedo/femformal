import numpy as np
from .util import project_list, project_regions, label_state, list_extr_points
import util
from .ts import abstract, state_n
from .modelcheck import check_spec

import logging
logger = logging.getLogger('FEMFORMAL')


def verify(system, partition, regions, init_states, spec, depth, **kwargs):
    if len(partition) != system.n:
        raise ValueError("System dimensions do not agree with partition")
    for key, item in regions.items():
        if len(item) != system.n:
            raise ValueError(
                "Region {0} dimensions do not agree with partition".format(key))
    dims = list(range(system.n))

    while len(dims) > 0:
        d = dims.pop(0)
        m = modelcheck(system, d, partition, regions, init_states, spec, depth)
        if m.sat:
            dims = [i for i in dims if i not in m.dchecked]
        else:
            return False
    return True


def verify_input_constrained(system, partition, regions, init_states, spec,
                             depth, **kwargs):
    if len(partition) != system.n:
        raise ValueError("System dimensions do not agree with partition")
    for key, item in regions.items():
        if len(item) != system.n:
            raise ValueError(
                "Region {0} dimensions do not agree with partition".format(key))
    dims = list(range(system.n))

    d = 1
    while d <= depth and d <= system.n:
        groups = make_groups(range(dims), d)
        subsl = [system.subsystem(g) for g in groups]
        p_partition_l = [project_list(partition, g) for g in groups]
        p_init_states_l = [[project_list(state, g) for state in init_states]
                           for g in groups]
        tsl = [abstract(subs, p_partition,
                    list_extr_points(project_list(
                        partition, system.pert_indices(indices))))
               for subs, p_partition in zip(subsl, p_partition_l)]
        initl = [[state_n(ts, state) for state in p_init_states]
                 for ts, p_init_states in zip(tsl, p_init_states_l)]

        verif_subsl = [Subsystem(*x) for x in zip(
            groups, subsl, p_partition_l, p_init_states_l, tsl, initl)]

        constrain_inputs(tsl, initl, p_partition_l)
        if all(check_spec(ts, spec, project_regions(regions, g), init)[0]
               for ts, g, init in zip(tsl, groups, initl)):
            return True
        else:
            d +=1

def constrain_inputs(subsystems, system):
    # f: subs x dindex ->  subs x index
    # [reach_sets]

    # FIXME add while
    ikeys = [min(subs.indices) for subs in subsystems]
    for subs in subsystems:
        pindices = system.pert_indices(subs.indices)
        subs.drelated = []
        for pi in pindices:
            si = bisect(ikeys, pi)
            subs_related = subsystems[si]
            subs.drelated.append((subs_related, system.pert_indices(
                subs_related.indices).index(pi)))

        #FIXME reach_set() to admit list of init states
        subs.reach_set = util.project_states(
            subs.ts.reach_set(subs.p_init_states))


    for subs in subsystems:
        pert_bounds = [[subs.p_pert_part[min(subsr.reach_set[pi])],
                        subs.p_pert_part[max(subsr.reach_set[pi]) + 1]]
                       for subsr, pi in subs.drelated]
        subs.ts = abstract(subs.subs, subs.p_part, pert_bounds)







class Subsystem(object):

    def __init__(self, indices, subs, p_part, p_init_states, ts, init):
        self.indices = indices
        self.subs = subs
        self.p_part = p_part
        self.p_init_states = p_init_states
        self.ts = ts
        self.init = init




def modelcheck(system, dim, partition, regions, init_states, spec, depth):
    indices = [dim]
    d = 1
    logger.debug(dim)

    while d <= depth and d <= system.n:
        logger.debug(indices)
        subs = system.subsystem(indices)
        p_partition = project_list(partition, indices)
        p_init_states = [project_list(state, indices) for state in init_states]
        ts = abstract(subs, p_partition,
                    list_extr_points(project_list(
                        partition, system.pert_indices(indices))))
        # util.draw_ts(ts)
        init = [state_n(ts, state) for state in p_init_states]
        sat, p = check_spec(ts, spec, project_regions(regions, indices), init)
        if sat:
            return ModelcheckResult(True, [])
        else:
            # l = random.choice(p)
            # s = label_state(l)
            # split_state()
            # d = len(s) - 1
            indices += system.pert_indices(indices)
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

