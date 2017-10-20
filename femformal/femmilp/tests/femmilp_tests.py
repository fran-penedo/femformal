import logging
import unittest

import numpy as np

from femformal.core import system as sys
from femformal.femmilp import femmilp as femmilp


logger = logging.getLogger(__name__)

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
        d_true, _ = sys.central_diff_integrate(sosys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)

    def test_fosys_trajectory(self):
        dt = 0.1
        fosys = sys.FOSystem(self.M, self.K, self.F, self.xpart, dt)
        d0 = np.array([1.0, 0.5, -0.5, -1.0])
        its = 100
        d = femmilp.simulate_trajectory(fosys, d0, its)
        d_true = sys.trapez_integrate(fosys, d0, its * dt, dt, alpha = 0)

        np.testing.assert_array_almost_equal(d, d_true)

    @unittest.skip("Long test for heat")
    def test_fosys_trajectory_heat(self):
        from examples.heat_mix.hm_synth_simple import cs
        its = 500
        f_nodal = np.zeros(cs.system.n)
        f_nodal[-1] = 1e6
        cs.system.F = f_nodal
        d = femmilp.simulate_trajectory(cs.system, cs.d0, its)
        d_true = sys.trapez_integrate(cs.system, cs.d0, its * cs.system.dt, cs.system.dt, alpha = 0)

        np.testing.assert_array_almost_equal(d, d_true)

    @unittest.skip("Long test for heat")
    def test_fosys_trajectory_heat_2(self):
        from examples.heat_mix.hm_synth_simple import cs
        its = 500
        T = its * cs.system.dt
        input_dt = 0.5
        inputs = [1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0]
        pwlf = sys.PWLFunction(np.linspace(0, T, round(T / input_dt) + 1), ys=inputs, x=100.0)
        def f_nodal(t):
            f = np.zeros(cs.system.n)
            f[-1] = pwlf(t, x=pwlf.x)
            return f


        csys = sys.ControlFOSystem.from_fosys(cs.system, f_nodal)
        d = femmilp.simulate_trajectory(csys, cs.d0, its)
        d_true = sys.trapez_integrate(csys, cs.d0, its * cs.system.dt, cs.system.dt, alpha = 0)

        np.testing.assert_array_almost_equal(d, d_true)

    def test_control_sosys_trajectory(self):
        dt = 0.1
        its = 100
        sosys = sys.SOSystem(self.M, self.K, self.F, self.xpart, dt)
        d0 = np.array([0.0, 0.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])
        dset = np.array([[1, 0], [-1, 0]])
        vset = np.array([[1, 0], [-1, 0]])
        fd = lambda x, p: p[0]
        fv = lambda x, p: p[0]
        inputs = [-5, 5, -5, 5]
        pwlf = sys.PWLFunction(np.linspace(0, dt * its, len(inputs)), inputs,
                               ybounds=[-10, 10], x=self.xpart[-1])
        fset = pwlf.pset()
        def f_nodal_control(t):
            f = np.zeros(4)
            f[-1] = pwlf(t, x=self.xpart[-1])
            return f
        csosys = sys.ControlSOSystem.from_sosys(sosys, f_nodal_control)

        d = femmilp.simulate_trajectory(sosys, None, its, [dset, vset, fset], [fd, fv, pwlf])
        d_true, _ = sys.central_diff_integrate(csosys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)


    def test_control_sosys_synth(self):
        dt = 0.1
        its = 100
        sosys = sys.SOSystem(self.M, self.K, self.F, self.xpart, dt)
        d0 = np.array([0.0, 0.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])
        dset = np.array([[1, 0], [-1, 0]])
        vset = np.array([[1, 0], [-1, 0]])
        fd = lambda x, p: p[0]
        fv = lambda x, p: p[0]
        pwlf1 = sys.PWLFunction(np.linspace(0, dt * its, 4),
                               ybounds=[-10, 10], x=self.xpart[-1])
        fset = pwlf1.pset()

        (_, inputs), d = femmilp.synthesize(
            sosys, [dset, vset, fset], [fd, fv, pwlf1], None, return_traj=True, T=its)
        pwlf2 = sys.PWLFunction(np.linspace(0, dt * its, len(inputs)), inputs,
                               ybounds=[-10, 10], x=self.xpart[-1])
        def f_nodal_control(t):
            f = np.zeros(4)
            f[-1] = pwlf2(t, x=self.xpart[-1])
            return f
        csosys = sys.ControlSOSystem.from_sosys(sosys, f_nodal_control)
        d_true, _ = sys.central_diff_integrate(csosys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)




    def test_hybsys_trajectory_simple(self):
        invariants = [(np.array([[1.0, 1.0]]), np.array([5e6])),
                      (-np.array([[1.0, 1.0]]), -np.array([5e6]))]
        values = [np.array([[1.0, -1.0],[-1.0, 1.0]]),
                  np.array([[1.0, -1.0], [-1.0, 1.0]]) * 0.5]

        K = [sys.HybridParameter(invariants, values) for i in range(3)]
        K[0].values = [np.identity(2) for i in range(2)]
        dt = 0.1
        its = 30
        hysys = sys.HybridSOSystem(self.M, K, self.F, self.xpart, dt)
        hysys.bigN = 1e7
        d0 = np.array([1.0, 0.5, -0.5, -1.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])

        d = femmilp.simulate_trajectory(hysys, [d0, v0], its)
        d_true, _ = sys.central_diff_integrate(hysys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)

    def test_hybsys_trajectory(self):
        invariants = [(np.array([[1.0, 1.0]]), np.array([5])),
                      (-np.array([[1.0, 1.0]]), -np.array([5]))]
        values = [np.array([[1.0, -1.0],[-1.0, 1.0]]),
                  np.array([[1.0, -1.0], [-1.0, 1.0]]) * 0.5]

        K = [sys.HybridParameter(invariants, values) for i in range(3)]
        K[0].values = [np.identity(2) for i in range(2)]
        dt = 0.1
        its = 30
        hysys = sys.HybridSOSystem(self.M, K, self.F, self.xpart, dt)
        hysys.bigN = 10000
        d0 = np.array([1.0, 0.5, -0.5, -1.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])

        d = femmilp.simulate_trajectory(hysys, [d0, v0], its)
        d_true, _ = sys.central_diff_integrate(hysys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d, d_true)
