import femformal.util as u
import numpy as np
import networkx as nx
import itertools as it

import logging
logger = logging.getLogger('FEMFORMAL')

def test_label():
    x = [1, -1, 2, 3]
    l_exp = 's1_-1_2_3'
    l = u.state_label(x)
    assert l == l_exp
    np.testing.assert_array_equal(u.label_state(l), x)

def make_groups_test():
    x = list(range(10))
    np.testing.assert_array_equal(u.make_groups(x, 1), [[i] for i in range(10)])
    np.testing.assert_array_equal(u.make_groups(x, 2),
                                  [[i, i+1] for i in range(0, 10, 2)])
    np.testing.assert_array_equal(u.make_groups(x, 3),
                                  [[0,1,2,3], [4,5,6],[7,8,9]])

def project_states_test():
    states = [[1,2,3], [4,3,2], [3,4,5]]
    np.testing.assert_array_equal([sorted(x) for x in u.project_states(states)],
                                  [[1,3,4], [2,3,4], [2,3,5]])

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

