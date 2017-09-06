import logging
import unittest
from itertools import product as cartesian_product
from nose.tools import nottest

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

    def test_interpolation_1dof_fixed_t(self):
        d = np.arange(self.mesh.nnodes)
        interp = self.mesh.interpolate(d)
        np.testing.assert_array_almost_equal(
            interp(2, 0.5), np.mean([0, 1, 5, 6]))
        for i in range(self.mesh.nnodes):
            np.testing.assert_array_almost_equal(
                interp(*self.mesh.nodes_coords[i]), d[i])

    def test_interpolation_1dof_mult_t(self):
        d = np.arange(self.mesh.nnodes)
        dd = np.array([d,d + 1])
        interp = self.mesh.interpolate(dd)
        np.testing.assert_array_almost_equal(
            interp(2, 0.5), [np.mean([0, 1, 5, 6]), np.mean([1, 2, 6, 7])])
        for i in range(self.mesh.nnodes):
            np.testing.assert_array_almost_equal(interp(*self.mesh.nodes_coords[i]), dd[:,i])

    def test_interpolation_2dof_mult_t(self):
        d = np.arange(self.mesh.nnodes * 2)
        dd = np.array([d,d + 1])
        logger.debug(dd.shape)
        interp = self.mesh.interpolate(dd)
        np.testing.assert_array_almost_equal(
            interp(2, 0.5),
            np.array([[np.mean([0, 2, 10, 12]), np.mean([1, 3, 11, 13])],
                      [np.mean([1, 3, 11, 13]), np.mean([2, 4, 12, 14])]]))
        for i in range(self.mesh.nnodes):
            np.testing.assert_array_almost_equal(
                interp(*self.mesh.nodes_coords[i]), dd[:, i*2:i*2+2].T)

    def test_interpolation_2dof_fixed_t(self):
        d = np.arange(self.mesh.nnodes)
        dd = np.array([d,d + 1])
        interp = self.mesh.interpolate(dd)
        np.testing.assert_array_almost_equal(
            interp(2, 0.5),
            np.array([np.mean([0, 1, 5, 6]), np.mean([1, 2, 6, 7])]))
        for i in range(self.mesh.nnodes):
            np.testing.assert_array_almost_equal(
                interp(*self.mesh.nodes_coords[i]), dd[:, i].T)


    @nottest
    def test_interpolation_derivatives(self):
        d = np.tile(np.arange(len(self.xs)), len(self.ys))
        interp = self.mesh.interpolate_derivatives(d)
        for i in range(self.mesh.nnodes):
            np.testing.assert_array_almost_equal(
                interp(*self.mesh.nodes_coords[i]),
                [0.25, 0])

