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
    partition = [[-1.0, 0.0, 1.0, 2.0, 3.0] for i in range(2)]
    regions = {'A': [1, 1]}
    spec = "G F state = A"
    init_states = [[3, 3]]

    assert v.verify(system, partition, regions, init_states, spec, 2) == True
