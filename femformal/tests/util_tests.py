import femformal.util as u
import numpy as np

import logging
logger = logging.getLogger('FEMFORMAL')

def labels_test():
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
