import logging

import numpy as np

from femformal.core import system as s, logic as logic


logger = logging.getLogger(__name__)

class test_verify(object):

    def test_verify_2d(self):
        A = np.array([[-5.0, 3.0], [-3.0, -5.0]])
        b = np.zeros((2, 1))
        C = np.empty(shape=(0,0))
        system = s.System(A, b, C)
        partition = [np.arange(-1.5, 3.5, 1).tolist() for i in range(2)]
        regions = {'A': [2, 2], 'B': [2, 3]}
        spec = "(X A) & (F (! (B)))"
        init_states = [[2, 3]]

        depth = 2

        # assert v.verify(system, partition, regions, init_states, spec, depth) == True

    def test_verify_ic_2d(self):
        A = np.array([[-5.0, 3.0], [-3.0, -5.0]])
        b = np.zeros((2, 1))
        C = np.empty(shape=(0,0))
        system = s.System(A, b, C)
        partition = [np.arange(-1.5, 3.5, 1).tolist() for i in range(2)]
        xpart = [1, 2]
        apcAl = logic.APCont(np.array([0, 2]), -1, lambda x: .5)
        apcAu = logic.APCont(np.array([0, 2]), 1, lambda x: 1.5)
        apcBl = logic.APCont(np.array([0, 2]), -1, lambda x: .5 if x==1 else 1.5)
        apcBu = logic.APCont(np.array([0, 2]), 1, lambda x: 1.5 if x==1 else 2.5)
        regions = {l : apd for (l, apd) in
                zip("ABCD",
                    [logic._ap_cont_to_disc(apc, xpart)
                        for apc in [apcAl, apcAu, apcBl, apcBu]])}
        spec = "(X (A & B)) & (F (! (C & D)))"
        init_states = [[2, 3]]

        depth = 2

        # assert v.verify_input_constrained(
        #     system, partition, regions, init_states, spec, depth, verbosity=10, draw_constr_ts=False) == True

