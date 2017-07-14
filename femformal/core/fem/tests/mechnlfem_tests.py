import unittest

import numpy as np

from femformal.core import system as sys
from femformal.core.fem import mechnlfem as mechnl, mechlinfem as mechlin


class TestMechNLFem(unittest.TestCase):
    def test_linear_equivalent(self):
        N = 5
        L = 100000
        rho_steel = 8e-6
        E_steel = 200e6
        sigma_yield_steel = 350e3
        yield_point_steel = sigma_yield_steel / E_steel
        E_steel_hybrid = sys.HybridParameter([
            lambda p: (np.array([[-1, 1]]) / np.diff(p), np.array([yield_point_steel])),
            lambda p: (np.array([[1, -1]]) / np.diff(p), np.array([-yield_point_steel]))],
            [E_steel, E_steel / 2.0])
        E_steel_hybrid.p = np.array([0, 1.0])

        self.assertEqual(E_steel_hybrid(np.array([0, 2*yield_point_steel])), E_steel / 2.0)
        self.assertEqual(E_steel_hybrid(np.array([0, .5*yield_point_steel])), E_steel)


        xpart = np.linspace(0, L, N + 1)
        g = [0.0, None]
        f_nodal = np.zeros(N + 1)
        dt = min((L / N) / np.sqrt(E_steel / rho_steel), (L / N) / np.sqrt(E_steel / 2 * rho_steel))

        nlsys = mechnl.mechnlfem(xpart, rho_steel, E_steel_hybrid, g, f_nodal, dt)
        lsys1 = mechlin.mechlinfem(xpart, rho_steel, E_steel, g, f_nodal, dt)
        lsys2 = mechlin.mechlinfem(xpart, rho_steel, E_steel / 2.0, g, f_nodal, dt)

        u0 = lambda x: 0.0
        du0 = lambda x: 0.0
        u1, v1 = mechnl.state(u0, du0, xpart, g)
        np.testing.assert_array_equal(nlsys.K(u1), lsys1.K)

        u0 = lambda x: 2 * yield_point_steel * x
        du0 = lambda x: 0.0
        u2, v2 = mechnl.state(u0, du0, xpart, g)
        np.testing.assert_array_equal(nlsys.K(u2), lsys2.K)

