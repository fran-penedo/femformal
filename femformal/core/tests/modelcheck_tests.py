import numpy as np
import femformal.femts.ts as t
import femformal.femts.modelcheck as m
from nose.tools import nottest

import logging
logger = logging.getLogger(__name__)

@nottest
def modelcheck_test():
    ts = t.TS()
    ts.add_nodes_from(['s1_1', 's2_2', 's3_3'])
    ts.add_edge('s1_1', 's2_2')
    ts.add_edge('s2_2', 's3_3')
    ts.add_edge('s3_3', 's3_3')
    spec = 'F G A'
    regions = {'A': [3, 3], 'B': [2, 2]}
    init = [0]

    sat, p = m.check_spec(ts, spec, regions, init)
    assert sat == True

    spec = 'F G B'
    sat, p = m.check_spec(ts, spec, regions, init)
    assert sat == False

