import logging
import unittest

import numpy as np

import femformal.core.system as sys
import femformal.femmilp.femmilp as femmilp


logger = logging.getLogger('FEMFORMAL')

class TestFemmilp(unittest.TestCase):
    def setUp(self):
        self.M = np.array([[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]) / 6.0
        self.K = np.array([[1.0, 0, 0, 0], [0, 2.0, -1.0, 0], [0, -1.0, 2.0, -1.0],
                      [0, 0, -1.0, 1.0]])
        self.F = np.array([0, 0, 0, 1.0])
        self.xpart = [0.0, 1.0, 2.0, 3.0]

    def test_sosys_trajectory(self):
        dt = 0.1
        sosys = sys.SOSystem(self.M, self.K, self.F, self.xpart, dt)
        d0 = np.array([1.0, 0.5, -0.5, -1.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])
        its = 100
        d = femmilp.simulate_trajectory(sosys, [d0, v0], its)
        d_true, _ = sys.newm_integrate(sosys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)

    def test_hybsys_trajectory_simple(self):
        invariants = [(np.array([[1.0, 1.0]]), np.array([5e6])),
                      (-np.array([[1.0, 1.0]]), -np.array([5e6]))]
        values = [np.array([[1.0, -1.0],[-1.0, 1.0]]),
                  np.array([[1.0, -1.0], [-1.0, 1.0]]) * 0.5]

        K = [sys.HybridParameter(invariants, values) for i in range(3)]
        K[0].values = [np.identity(2) for i in range(2)]
        dt = 0.1
        its = 100
        hysys = sys.HybridSOSystem(self.M, K, self.F, self.xpart, dt)
        hysys.bigN = 1e7
        d0 = np.array([1.0, 0.5, -0.5, -1.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])

        d = femmilp.simulate_trajectory(hysys, [d0, v0], its)
        d_true, _ = sys.newm_integrate(hysys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)


    def test_hybsys_trajectory(self):
        invariants = [(np.array([[1.0, 1.0]]), np.array([5])),
                      (-np.array([[1.0, 1.0]]), -np.array([5]))]
        values = [np.array([[1.0, -1.0],[-1.0, 1.0]]),
                  np.array([[1.0, -1.0], [-1.0, 1.0]]) * 0.5]

        K = [sys.HybridParameter(invariants, values) for i in range(3)]
        K[0].values = [np.identity(2) for i in range(2)]
        dt = 0.1
        its = 100
        hysys = sys.HybridSOSystem(self.M, K, self.F, self.xpart, dt)
        hysys.bigN = 10000
        d0 = np.array([1.0, 0.5, -0.5, -1.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])

        d = femmilp.simulate_trajectory(hysys, [d0, v0], its)
        d_true, _ = sys.newm_integrate(hysys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)

