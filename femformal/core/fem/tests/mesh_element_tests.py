import logging
import unittest
from itertools import product as cartesian_product

import numpy as np

from .. import mesh
from .. import element


logger = logging.getLogger(__name__)

class TestGridQ4BLQuad(unittest.TestCase):
    def setUp(self):
        self.L = 16
        self.c = 2
        self.xs = np.linspace(0, self.L, 5)
        self.ys = np.linspace(0, self.c, 3)
        self.mesh = mesh.GridQ4([self.xs, self.ys], element.BLQuadQ4)

    def test_interpolation(self):
        d = np.arange(self.mesh.nnodes)
        dd = np.array([d,d + 1]).T
        logger.debug(dd)
        interp = self.mesh.interpolate(dd)
        np.testing.assert_array_almost_equal(
            interp(2, 0.5), [np.mean([0, 1, 5, 6]), np.mean([1, 2, 6, 7])])




