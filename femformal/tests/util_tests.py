import femformal.util as u
import numpy as np
import networkx as nx
import itertools as it

import logging
logger = logging.getLogger('FEMFORMAL')

def labels_test():
    x = [1, -1, 2, 3]
    l_exp = 's1_-1_2_3'
    l = u.state_label(x)
    assert l == l_exp
    np.testing.assert_array_equal(u.label_state(l), x)


def draw_ts_2D_test():
    partition = [list(range(5)) for i in [0,1]]
    ts = nx.DiGraph()
    indices = [list(range(len(p) - 1)) for p in partition]
    states = list(it.product(*indices))
    ts.add_nodes_from([u.state_label(s) for s in states])
    ts.add_edge(u.state_label([2,2]), u.state_label([3,2]))
    ts.add_edge(u.state_label([2,2]), u.state_label([2,3]))
    ts.add_edge(u.state_label([2,2]), u.state_label([1,2]))
    ts.add_edge(u.state_label([2,2]), u.state_label([2,1]))
    ts.add_edge(u.state_label([3,2]), u.state_label([2,2]))
    ts.add_edge(u.state_label([2,2]), u.state_label([2,2]))

    # u.draw_ts_2D(ts, partition)


