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

