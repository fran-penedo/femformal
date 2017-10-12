import logging
import unittest

import numpy as np

from .. import system as sys
from ..fem import mesh, element


logger = logging.getLogger(__name__)

class TestSystem(unittest.TestCase):

    def test_system(self):
        A = np.eye(5,5)*4 + np.eye(5,5,k=-1)*3 + np.eye(5,5,k=1)*2
        b = np.array([1,2,3,4,5])
        b = b[:,None]
        i = [0,1,4]
        Ai = np.array([[4, 2, 0], [3, 4, 0], [0, 0, 4]])
        bi = np.array([1,2,5])
        bi = bi[:, None]
        Ci = np.array([[0, 0], [2, 0], [0, 3]])

        S = sys.System(A, b)

        assert S.n == 5
        assert S.m == 0

        Ss = S.subsystem(i)

        np.testing.assert_array_equal(Ss.A, Ai)
        np.testing.assert_array_equal(Ss.b, bi)
        np.testing.assert_array_equal(Ss.C, Ci)

    def test_reach(self):
        A = np.array([[-2, 1], [1, -2]])
        b = np.zeros((2, 1))
        C = np.empty(shape=(0,0))
        system = sys.System(A, b, C)
        dist_bounds = np.empty(shape=(0,0))
        R1 = np.array([[-1, 1], [-1, 1]])
        for i in range(2):
            # logger.debug(i)
            facet = R1.copy()
            facet[i, 0] = facet[i, 1]
            assert sys.is_facet_separating(system, facet, 1, i, dist_bounds)
            facet = R1.copy()
            facet[i, 1] = facet[i, 0]
            # logger.debug(facet)
            assert sys.is_facet_separating(system, facet, -1, i, dist_bounds)

        facet = np.array([[-1, 1], [-0.1, -0.1]])
        assert not sys.is_facet_separating(system, facet, 1, 0, dist_bounds)

    def test_reach_facet(self):
        A = np.array([[-2, 1], [1, -2]])
        b = np.zeros((2, 1))
        C = np.empty(shape=(0,0))
        system = sys.System(A, b, C)
        dist_bounds = np.empty(shape=(0,0))
        R1 = np.array([[-1, -1], [-1, 1]])
        assert sys.is_facet_separating(system, R1, -1, 0, dist_bounds)

        R2 = np.array([[-1, -1], [1, 2]])
        assert not sys.is_facet_separating(system, R2, 1, 0, dist_bounds)

    def test_region_invariant(self):
        A = np.array([[-2, 1], [1, -2]])
        b = np.zeros((2, 1))
        C = np.empty(shape=(0,0))
        system = sys.System(A, b, C)
        dist_bounds = np.empty(shape=(0,0))
        R = np.array([[-1, 1], [-1, 1]])
        assert sys.is_region_invariant(system, R, dist_bounds)
        R = np.array([[-1, -0.2], [-1, 1]])
        assert not sys.is_region_invariant(system, R, dist_bounds)


class TestComplexSystem(unittest.TestCase):
    def setUp(self):
        M = np.array([[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]) / 6.0
        K = np.array([[1.0, 0, 0, 0], [0, 1.0, -1.0, 0], [0, -1.0, 1.0, -1.0],
                      [0, 0, -1.0, 1.0]])
        F = np.array([0, 0, 0, 1.0])
        dt = .1
        xpart = [0.0, 1.0, 2.0, 3.0]
        self.sosys = sys.SOSystem(M, K, F, xpart, dt)
        self.pwlf = sys.PWLFunction([0, 1, 2, 3], [0, 1.0, 0.5, 2.0], ybounds=[-10.0, 10.0])

    def test_cont_disc(self):
        A = np.array([[-2, 1], [1, -2]])
        b = np.array([[1], [2]])
        system = sys.System(A, b)
        x0 = np.array([1,2])
        t_cont = np.linspace(0, 10, 101)
        t_disc = 10

        system_d = sys.cont_to_disc(system)
        x_c = sys.cont_integrate(system, x0, t_cont)
        x_d = sys.disc_integrate(system_d, x0, t_disc)
        np.testing.assert_array_almost_equal(x_c[-1], x_d[-1], decimal=1)

        t_cont = np.linspace(0, 1, 101)
        t_disc = 20

        system_d = sys.cont_to_disc(system, .5)
        x_c = sys.cont_integrate(system, x0, t_cont)
        x_d = sys.disc_integrate(system_d, x0, t_disc)
        np.testing.assert_array_almost_equal(x_c[-1], x_d[-1], decimal=1)

    def test_newmark(self):
        sosys = self.sosys
        fosys = sosys.to_fosystem().to_canon()
        dt = sosys.dt
        T = 10
        t_cont = np.linspace(0, T, int(T/dt) + 1)

        d0 = [0.0 for i in range(sosys.n)]
        v0 = [0.0 for i in range(sosys.n)]

        d_so, v_so = sys.newm_integrate(sosys, d0, v0, T, dt)
        y_fo = sys.cont_integrate(fosys, d0 + v0, t_cont)

        np.testing.assert_array_almost_equal(d_so[10], y_fo[10,0:sosys.n], decimal=2)

    def test_newmark_constant(self):
        """Test a constant second order system"""

        xpart = np.linspace(0, 10, 6)
        xn = len(xpart)
        xsys = sys.SOSystem(
            np.zeros((xn, xn)), np.identity(xn), np.zeros(xn), dt=1.0, xpart=xpart)
        x, _ = sys.newm_integrate(
            xsys, 1 + np.zeros(xn), np.zeros(xn), 10, xsys.dt)
        np.testing.assert_array_equal(x.shape, (11, xn))
        np.testing.assert_equal(np.all(np.isclose(x, 1)), True)


    def test_pwlf(self):
        self.assertEquals(self.pwlf.ys, [self.pwlf(t) for t in self.pwlf.ts])
        self.assertEquals(self.pwlf(1.5), 0.75)

    def test_pwlf_disc(self):
        pwlf = sys.PWLFunction([0, 1, 1, 2], [0, 0, 1.0, 1.0])
        self.assertEquals([0, 1.0, 1.0, 1.0], [pwlf(t) for t in pwlf.ts])
        self.assertEquals(pwlf(1.5), 1.0)
        self.assertEquals(pwlf(0.5), 0.0)

    def test_pwlf_disc_high_to_low(self):
        pwlf = sys.PWLFunction([0, 1, 1, 2], [1.0, 1.0, 0, 0])
        self.assertEquals([1.0, 0, 0, 0], [pwlf(t) for t in pwlf.ts])
        self.assertEquals(pwlf(1.5), 0)
        self.assertEquals(pwlf(0.5), 1.0)
        self.assertEquals(pwlf(0.999), 1.0)

    def test_pwlf_pset(self):
        pset = np.array([[1, 0, 0, 0, 10.0],
                         [0, 1, 0, 0, 10.0],
                         [0, 0, 1, 0, 10.0],
                         [0, 0, 0, 1, 10.0],
                         [-1, 0, 0, 0, 10.0],
                         [0, -1, 0, 0, 10.0],
                         [0, 0, -1, 0, 10.0],
                         [0, 0, 0, -1, 10.0]])
        np.testing.assert_array_equal(self.pwlf.pset(), pset)

    def test_control(self):
        pwlf = sys.PWLFunction([0, 1, 1, 2], [1.0, 1.0, 0.0, 0.0])
        sosys = self.sosys
        dt = sosys.dt

        d0 = [0.0 for i in range(sosys.n)]
        v0 = [0.0 for i in range(sosys.n)]

        sosys.F = np.array([0, 0, 0, 0])
        csys = sys.ControlSOSystem.from_sosys(
            sosys, lambda t: np.r_[np.zeros(sosys.n - 1), pwlf(t)])
        dc, vc = sys.newm_integrate(csys, d0, v0, 2, dt)

        sosys.F = np.array([0, 0, 0, 1.0])
        d, v = sys.newm_integrate(sosys, d0, v0, 1, dt)
        sosys.F = np.array([0, 0, 0, 0.0])
        # At t = 1, a must be computed using F = 0, so v[-1] is incorrect.
        # Can't stop at t = 0.9 and change F, since at t=0.9, F = 1.0 still
        d2, v2 = sys.newm_integrate(sosys, d[-1], vc[int(1/dt)], 1, dt)

        np.testing.assert_array_equal(dc[:int(1/dt) + 1], d)
        np.testing.assert_array_equal(vc[:int(1/dt)], v[:-1])
        np.testing.assert_array_equal(dc[int(1/dt):], d2)
        np.testing.assert_array_equal(vc[int(1/dt):], v2)


class TestHybridSystem(unittest.TestCase):
    def setUp(self):
        self.M = np.array([[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]) / 6.0
        self.K = np.array([[1.0, 0, 0, 0], [0, 2.0, -1.0, 0], [0, -1.0, 2.0, -1.0],
                      [0, 0, -1.0, 1.0]])
        self.F = np.array([0, 0, 0, 1.0])
        self.xpart = [0.0, 1.0, 2.0, 3.0]

    def test_matrix_function(self):
        f = lambda u: np.arange(25).reshape(5,5)
        mf = sys._MatrixFunction(f)
        u = 10
        np.testing.assert_array_equal(mf(u), f(u))
        np.testing.assert_array_equal(mf[1:3, 2:4](u), f(u)[1:3, 2:4])
        np.testing.assert_array_equal(mf[1:3, 2:4][1,1](u), f(u)[1:3, 2:4][1,1])

    def test_hybrid_k_global(self):
        M = np.identity(3)
        K = [lambda u, i=i: (i+1) * np.array([[u[0], -u[0]], [-u[1], u[1]]]) for i in range(2)]
        F = np.zeros(3)
        hsos = sys.HybridSOSystem(M, K, F)
        np.testing.assert_array_equal(
            hsos.K_global([2,3,4]),
            np.array([[2, -2, 0], [-3, 9, -6], [0, -8, 8]]))

    def test_hybrid_simple(self):
        M = np.identity(2)
        K = [lambda u: np.zeros((2,2)) if u[0] < 0.5 else np.identity(2)]
        F = np.array([1,1])
        hsos = sys.HybridSOSystem(M, K, F)
        d0 = [0.0, 0.0]
        v0 = [0.0, 0.0]
        dt = 0.1

        d, v = sys.newm_integrate(hsos, d0, v0, 2, dt)
        K1 = np.zeros((2,2))
        K2 = np.identity(2)
        sos1 = sys.SOSystem(M, K1, F)
        sos2 = sys.SOSystem(M, K2, F)
        d1, v1 = sys.newm_integrate(sos1, d0, v0, 1, dt)
        d2, v2 = sys.newm_integrate(sos2, d1[-1], v[int(1/dt)], 1, dt)

        np.testing.assert_array_equal(d[:int(1/dt) + 1], d1)
        np.testing.assert_array_equal(v[:int(1/dt)], v1[:-1])
        np.testing.assert_array_equal(d[int(1/dt):], d2)
        np.testing.assert_array_equal(v[int(1/dt):], v2)

    def test_hybrid_parameter(self):
        par = sys.HybridParameter([
            lambda p: (np.array([[1, -1]]) / np.diff(p), np.array([1])),
            lambda p: (np.array([[-1, 1]]) / np.diff(p), np.array([-1]))],
            [1, -1], p=[2.0, 1.0])

        self.assertEqual(par(np.array([3, 1])), 1)
        self.assertEqual(par(np.array([-1, 1])), -1)
        self.assertEqual(par(np.array([2, 1])), 1)

        invs = par.invariants
        reprs = par.invariant_representatives()
        for inv, rep in zip(invs, reprs):
            self.assertTrue(np.all(inv[0].dot(rep) <= inv[1]))

    def test_hybrid_system_trajectory_nonhybrid(self):
        invariants = [(np.array([[1.0, 1.0]]), np.array([5e6])),
                      (-np.array([[1.0, 1.0]]), -np.array([5e6]))]
        values = [np.array([[1.0, -1.0],[-1.0, 1.0]]),
                  np.array([[1.0, -1.0], [-1.0, 1.0]]) * 0.5]

        K = [sys.HybridParameter(invariants, values) for i in range(3)]
        K[0].values = [np.identity(2) for i in range(2)]
        dt = 0.1
        its = 100
        hysys = sys.HybridSOSystem(self.M, K, self.F, self.xpart, dt)
        sosys = sys.SOSystem(self.M, self.K, self.F, self.xpart, dt)
        d0 = np.array([1.0, 0.5, -0.5, -1.0])
        v0 = np.array([0.0, 0.0, 0.0, 0.0])

        np.testing.assert_array_almost_equal(hysys.K_global(d0), sosys.K)
        d_sosys, _ = sys.newm_integrate(sosys, d0, v0, its * dt, dt)
        d_hyb, _ = sys.newm_integrate(hysys, d0, v0, its * dt, dt)

        np.testing.assert_array_almost_equal(d_hyb, d_sosys)


class TestSystemDiff2D1DOF(unittest.TestCase):
    def setUp(self):
        xxs = np.linspace(0, 16, 5)
        yxs = np.linspace(0, 16, 9)
        xys = np.linspace(0, 2, 3)
        yys = np.linspace(0, 2, 5)
        self.xmesh = mesh.GridQ4([xxs, xys], element.BLQuadQ4)
        self.ymesh = mesh.GridQ4([yxs, yys], element.BLQuadQ4)
        xn, yn = self.xmesh.nnodes, self.ymesh.nnodes
        self.T = 10
        self.xsys = sys.SOSystem(
            np.zeros((xn, xn)), np.identity(xn), np.zeros(xn), dt=2.0, mesh=self.xmesh)
        self.ysys = sys.SOSystem(
            np.zeros((yn, yn)), np.identity(yn), np.zeros(yn), dt=1.0, mesh=self.ymesh)
        self.x0 = [1 + np.zeros(xn), np.zeros(xn)]
        self.y0 = [2 + np.zeros(yn), np.zeros(yn)]
        self.x, _ = sys.newm_integrate(
            self.xsys, self.x0[0], self.x0[1], self.T, self.xsys.dt)
        self.y, _ = sys.newm_integrate(
            self.ysys, self.y0[0], self.y0[1], self.T, self.ysys.dt)
        self.xlims = np.array([[4, 0], [12, 2]])


    def test_diff2d(self):
        xl, xr = self.xlims
        diff = sys.diff2d(
            self.x, self.y, self.xsys.dt, self.ysys.dt, self.xmesh, self.ymesh, xl, xr)
        np.testing.assert_array_equal(
            diff.shape, (self.ymesh.nnodes - 4*5, self.x.shape[0]))
        np.testing.assert_equal(np.all(np.isclose(diff, -1)), True)

    def test_sosys_diff_2d(self):
        np.testing.assert_array_almost_equal(
            sys.sosys_diff(self.xsys, self.ysys, self.x0, self.y0, [0, self.T], self.xlims),
            1 + np.zeros((self.ymesh.nnodes - 4*5, 1 + int(round(self.T / self.xsys.dt)))))

    def test_sosys_max_der_diff_2d(self):
        d0 = np.tile(np.arange(5), 3)
        v0 = np.zeros(self.xmesh.nnodes)
        mdx, mdtx = sys.sosys_max_der_diff(self.xsys, [d0, v0], [0, self.T])
        np.testing.assert_array_equal(mdx.shape, (self.xmesh.nelems, 2, 2))
        np.testing.assert_array_almost_equal(
            mdx, np.tile([[0.25, 0], [0, 0]], (self.xmesh.nelems, 1, 1)))
        np.testing.assert_array_equal(mdtx.shape, (self.xmesh.nnodes, ))


class TestSystemDiff2D2DOF(unittest.TestCase):
    def setUp(self):
        xxs = np.linspace(0, 16, 5)
        yxs = np.linspace(0, 16, 9)
        xys = np.linspace(0, 2, 3)
        yys = np.linspace(0, 2, 5)
        self.xmesh = mesh.GridQ4([xxs, xys], element.BLQuadQ4)
        self.ymesh = mesh.GridQ4([yxs, yys], element.BLQuadQ4)
        xn, yn = 2 * self.xmesh.nnodes, 2 * self.ymesh.nnodes
        self.T = 10
        self.xsys = sys.SOSystem(
            np.zeros((xn, xn)), np.identity(xn), np.ones(xn), dt=2.0, mesh=self.xmesh)
        self.ysys = sys.SOSystem(
            np.zeros((yn, yn)), np.identity(yn), np.ones(yn), dt=1.0, mesh=self.ymesh)
        self.x0 = [1 + np.zeros(xn), np.zeros(xn)]
        self.y0 = [2 + np.zeros(yn), np.zeros(yn)]
        self.x, _ = sys.newm_integrate(
            self.xsys, self.x0[0], self.x0[1], self.T, self.xsys.dt)
        self.y, _ = sys.newm_integrate(
            self.ysys, self.y0[0], self.y0[1], self.T, self.ysys.dt)
        self.xlims = np.array([[4, 0], [12, 2]])


    def test_diff2d(self):
        xl, xr = self.xlims
        diff = sys.diff2d(
            self.x, self.y, self.xsys.dt, self.ysys.dt, self.xmesh, self.ymesh, xl, xr)
        np.testing.assert_array_equal(
            diff.shape, (self.ymesh.nnodes - 4*5, 2, self.x.shape[0]))
        np.testing.assert_equal(np.all(np.isclose(diff, -1)), True)

    def test_sosys_diff_2d(self):
        np.testing.assert_array_almost_equal(
            sys.sosys_diff(self.xsys, self.ysys, self.x0, self.y0, [0, self.T], self.xlims),
            1 + np.zeros((self.ymesh.nnodes - 4*5, 2, 1 + int(round(self.T / self.xsys.dt)))))

    def test_sosys_max_der_diff_2d(self):
        d0 = np.zeros(2 * self.xmesh.nnodes)
        d0[::2] = np.tile(np.arange(5), 3)
        d0[1::2] = 1 + np.tile(np.arange(5), 3)
        v0 = np.zeros(2 * self.xmesh.nnodes)
        mdx, mdtx = sys.sosys_max_der_diff(self.xsys, [d0, v0], [0, self.T])
        np.testing.assert_array_equal(mdx.shape, (self.xmesh.nelems, 2, 2))
        np.testing.assert_array_almost_equal(
            mdx, np.tile([[0.25, 0], [0.25, 0]], (self.xmesh.nelems, 1, 1)))
        np.testing.assert_array_equal(mdtx.shape, (2 * self.xmesh.nnodes, ))
