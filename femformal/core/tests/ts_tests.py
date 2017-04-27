import numpy as np
import networkx as nx
import femformal.femts.ts as t
import femformal.core.system as s
from femformal.core.draw_util import draw_ts

import logging
logger = logging.getLogger('FEMFORMAL')

class test_ts(object):

    def test_reach_set(self):
        ts = t.TS()
        ts.add_edge(1,1)
        ts.add_edge(1,2)
        ts.add_edge(3, 2)
        ts.add_edge(1,5)
        ts.add_edge(5, 3)
        ts.add_edge(6,3)
        ts.add_edge(6,6)

        np.testing.assert_array_equal(sorted(ts.reach_set([1])), [1,2,3,5])


    def test_state_n(self):
        ts = t.TS()
        ts.add_nodes_from(['s1_1', 's2_2', 's3_3'])
        ts.add_edge('s1_1', 's2_2')
        ts.add_edge('s2_2', 's3_3')
        ts.add_edge('s3_3', 's3_3')

        n = t.state_n(ts, [2, 2])
        assert ts.nodes()[n] == 's2_2'

    def test_abstract(self):
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        b = np.zeros((2, 1))
        C = np.empty(shape=(0,0))
        system = s.System(A, b, C)
        partition = [[-2.0, -1.0, 1.0, 2.0] for i in range(2)]
        dist_bounds = []

        ts = t.abstract(system, partition, dist_bounds)
        # logger.debug(ts.nodes(data=True))

        # draw_ts(ts)


    def test_abstract2(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.zeros((2, 1))
        C = np.empty(shape=(0,0))
        system = s.System(A, b, C)
        partition = [[-2.0, -1.0, 1.0, 2.0] for i in range(2)]
        dist_bounds = []

        ts = t.abstract(system, partition, dist_bounds)
        # logger.debug(ts.nodes(data=True))

        # draw_ts(ts)


