import femformal.verify as v
import femformal.system as s
import numpy as np

import logging
logger = logging.getLogger('FEMFORMAL')

def verify_2d_test():
    A = np.array([[-3.0, 2.0], [2.0, -3.0]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    partition = [np.arange(-1, 3.5, 0.5).tolist() for i in range(2)]
    regions = {'A': [2, 2]}
    spec = "F G state = A"
    init_states = [[6, 6]]

    depth = 2

    assert v.verify(system, partition, regions, init_states, spec, depth) == True

