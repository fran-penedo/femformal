import unittest
import logging

import numpy as np

from .. import element as element


logger = logging.getLogger(__name__)

class TestBLQuadQ4(unittest.TestCase):
    def setUp(self):
        self.elem = element.BLQuadQ4(np.array([[-1.0, -1], [1, -1], [1, 2], [-1, 2]]))


    def test_shapes(self):
        pts = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1]])
        expected = np.identity(pts.shape[0])
        for i in range(pts.shape[0]):
            np.testing.assert_array_equal(self.elem.shapes(*pts[i, :]), expected[i])
        np.testing.assert_array_equal(self.elem.shapes(0, 0),
                                      np.array([0.25, 0.25, 0.25, 0.25]))

    def test_shapes_derivatives(self):
        pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
        expected = [[np.array([[-0.3943, 0.3943, 0.1057, -0.1057],
                               [-0.3943, -0.1057, 0.1057, 0.3943]]),
                     np.array([[-0.1057, 0.1057, 0.3943, -0.3943],
                               [-0.3943, -0.1057, 0.1057, 0.3943]])],
                    [np.array([[-0.3943, 0.3943, 0.1057, -0.1057],
                               [-0.1057, -0.3943, 0.3943, 0.1057]]),
                     np.array([[-0.1057, 0.1057, 0.3943, -0.3943],
                               [-0.1057, -0.3943, 0.3943, 0.1057]])]]

        for i in range(len(pts)):
            for j in range(len(pts)):
                np.testing.assert_array_almost_equal(
                    self.elem.shapes_derivatives(pts[i], pts[j]),
                    expected[i][j], decimal=4)


    def test_normalize(self):
        test = np.vstack([self.elem.coords, self.elem.center])
        expected = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])
        for t, e in zip(test, expected):
            np.testing.assert_array_equal(self.elem.normalize(t), e)

    def test_interpolate(self):
        vs = np.array([1,2,3,4])
        np.testing.assert_equal(self.elem.interpolate(vs, [0, 0]), np.mean(vs))

    def test_interpolate_phys(self):
        vs = np.array([1,2,3,4])
        np.testing.assert_equal(self.elem.interpolate_phys(vs, [0, 0.5]), np.mean(vs))
