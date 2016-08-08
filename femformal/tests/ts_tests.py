import numpy as np
import networkx as nx
import femformal.ts as t
import femformal.system as s
from femformal.util import draw_ts

import logging
logger = logging.getLogger('FEMFORMAL')

def abstract_test():
    A = np.array([[-2, 1], [1, -2]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    partition = [[-2, -1, 1, 2] for i in range(2)]
    dist_bounds = []

    ts = t.abstract(system, partition, dist_bounds)
    logger.debug(ts.nodes(data=True))

    draw_ts(ts)


