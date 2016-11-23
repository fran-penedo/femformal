import femformal.logic as logic
import numpy as np
import itertools as it

def _ap_test():
    apc = logic.APCont(np.array([0, 2]), 1, lambda x: x)
    xpart = [0,1,2,3]
    apd = logic.ap_cont_to_disc(apc, xpart)
    assert apd.r == 1
    assert apd.m[0] == 1
    assert apd.m[1] == 2
    assert 2 not in apd.m

    tpart = [0.5, 1, 1.5, 2.5, 3]
    proj = logic.project_apdisc(apd, [0,1], tpart)
    np.testing.assert_array_equal(proj, list(it.product(*[[0], [0, 1]])))
    apd.r = -1
    proj = logic.project_apdisc(apd, [0,1], tpart)
    np.testing.assert_array_equal(proj, list(it.product(*[[1,2,3], [3]])))
