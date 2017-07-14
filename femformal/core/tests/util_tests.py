import itertools as it
import logging

import networkx as nx
import numpy as np

from femformal.core import util as u


logger = logging.getLogger(__name__)

class test_util(object):
    def test_state_label(self):
        x = [1, -1, 2, 3]
        l_exp = 's1_-1_2_3'
        l = u.state_label(x)
        assert l == l_exp
        np.testing.assert_array_equal(u.label_state(l), x)


    def test_label(self):
        l = "x"
        i = 3
        j = 14
        label = u.label(l, i, j)
        assert label == "x_3_14"
        ll, ii, jj = u.unlabel(label)
        assert i == ii
        assert j == jj
        assert l == ll


    def test_make_groups(self):
        x = list(range(10))
        np.testing.assert_array_equal(u.make_groups(x, 1), [[i] for i in range(10)])
        np.testing.assert_array_equal(u.make_groups(x, 2),
                                    [[i, i+1] for i in range(0, 10, 2)])
        np.testing.assert_array_equal(u.make_groups(x, 3),
                                    [[0,1,2,3], [4,5,6],[7,8,9]])

    def test_project_states(self):
        states = [[1,2,3], [4,3,2], [3,4,5]]
        np.testing.assert_array_equal([sorted(x) for x in u.project_states(states)],
                                    [[1,3,4], [2,3,4], [2,3,5]])

    def test_draw_ts_2D(self):
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

