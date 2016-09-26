import femformal.verify as v
import femformal.system as s
import numpy as np

import logging
logger = logging.getLogger('FEMFORMAL')

def verify_2d_test():
    A = np.array([[-5.0, 3.0], [-3.0, -5.0]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    partition = [np.arange(-1.5, 3.5, 1).tolist() for i in range(2)]
    regions = {'A': [2, 2], 'B': [2, 3]}
    spec = "(X state = A) & (F (! (state = B)))"
    init_states = [[2, 3]]

    depth = 2

    assert v.verify(system, partition, regions, init_states, spec, depth) == True

def verify_ic_2d_test():
    A = np.array([[-5.0, 3.0], [-3.0, -5.0]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    partition = [np.arange(-1.5, 3.5, 1).tolist() for i in range(2)]
    regions = {'A': [2, 2], 'B': [2, 3]}
    spec = "(X state = A) & (F (! (state = B)))"
    init_states = [[2, 3]]

    depth = 2

    assert v.verify_input_constrained(
        system, partition, regions, init_states, spec, depth) == True

