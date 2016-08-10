import numpy as np
import networkx as nx
import femformal.ts as t
import femformal.system as s
from femformal.util import draw_ts

import logging
logger = logging.getLogger('FEMFORMAL')

def abstract_test():
    A = np.array([[-2.0, 1.0], [1.0, -2.0]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    partition = [[-2.0, -1.0, 1.0, 2.0] for i in range(2)]
    dist_bounds = []

    ts = t.abstract(system, partition, dist_bounds)
    logger.debug(ts.nodes(data=True))

    # draw_ts(ts)


def abstract2_test():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    partition = [[-2.0, -1.0, 1.0, 2.0] for i in range(2)]
    dist_bounds = []

    ts = t.abstract(system, partition, dist_bounds)
    logger.debug(ts.nodes(data=True))

    # draw_ts(ts)


def modelcheck_test():
    ts = t.TS()
    ts.add_nodes_from(['s1_1', 's2_2', 's3_3'])
    ts.add_edge('s1_1', 's2_2')
    ts.add_edge('s2_2', 's3_3')
    ts.add_edge('s3_3', 's3_3')
    spec = 'F G state = A'
    regions = {'A': [3, 3], 'B': [2, 2]}
    init = [0]

    sat, p = ts.modelcheck(spec, regions, init)
    assert sat == True

    spec = 'F G state = B'
    sat, p = ts.modelcheck(spec, regions, init)
    assert sat == False

