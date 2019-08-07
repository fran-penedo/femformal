import itertools as it
import logging

import numpy as np

from femformal.core import util as u


logger = logging.getLogger(__name__)


class test_util(object):
    def test_state_label(self):
        x = [1, -1, 2, 3]
        l_exp = "s1_-1_2_3"
        l = u.state_label(x)
        assert l == l_exp
        np.testing.assert_array_equal(u.label_state(l), x)

    def test_label(self):
        l = "x"
        i = 3
        j = 14
        label = u.label(l, i, j)
        assert label == "x_3_14"
        ll, ii, jj = u.unlabel(label)
        assert i == ii
        assert j == jj
        assert l == ll

    def test_make_groups(self):
        x = list(range(10))
        np.testing.assert_array_equal(u.make_groups(x, 1), [[i] for i in range(10)])
        np.testing.assert_array_equal(
            u.make_groups(x, 2), [[i, i + 1] for i in range(0, 10, 2)]
        )
        np.testing.assert_array_equal(
            u.make_groups(x, 3), [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
        )

    def test_project_states(self):
        states = [[1, 2, 3], [4, 3, 2], [3, 4, 5]]
        np.testing.assert_array_equal(
            [sorted(x) for x in u.project_states(states)],
            [[1, 3, 4], [2, 3, 4], [2, 3, 5]],
        )

    def test_cycle(self):
        expected = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        for actual, exp in it.izip(u.cycle(3), expected):
            np.testing.assert_array_equal(list(actual), exp)
