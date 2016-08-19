import numpy as np
import femformal.ts as t
import femformal.modelcheck as m

import logging
logger = logging.getLogger('FEMFORMAL')

def modelcheck_test():
    ts = t.TS()
    ts.add_nodes_from(['s1_1', 's2_2', 's3_3'])
    ts.add_edge('s1_1', 's2_2')
    ts.add_edge('s2_2', 's3_3')
    ts.add_edge('s3_3', 's3_3')
    spec = 'F G state = A'
    regions = {'A': [3, 3], 'B': [2, 2]}
    init = [0]

    sat, p = m.check_spec(ts, spec, regions, init)
    assert sat == True

    spec = 'F G state = B'
    sat, p = m.check_spec(ts, spec, regions, init)
    assert sat == False

