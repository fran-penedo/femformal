import unittest
from os import path

import numpy as np
import numpy.testing as npt
from scipy import io

from .. import mech2d as mech2d
from .. import element as element


class TestMech2d(unittest.TestCase):
    def setUp(self):
        self.C = np.array(
            [
                [1.346153846153846e07, 5.769230769230769e06, 0.000000000000000e00],
                [5.769230769230769e06, 1.346153846153846e07, 0.000000000000000e00],
                [0.000000000000000e00, 0.000000000000000e00, 3.846153846153846e06],
            ]
        )
        self.x = np.array([12, 16, 16, 12])
        self.y = np.array([1.0, 1.0, 2.0, 2.0])
        self.L = 16
        self.c = 2
        self.xs = np.linspace(0, self.L, 5)
        self.ys = np.linspace(0, self.c, 3)
        self.rho = 0.5

    def test_elem_stiffness(self):
        expected = np.array(
            [
                [
                    6.250000000000003e06,
                    2.403846153846154e06,
                    1.442307692307694e06,
                    4.807692307692308e05,
                    -3.125000000000001e06,
                    -2.403846153846154e06,
                    -4.567307692307697e06,
                    -4.807692307692308e05,
                ],
                [
                    2.403846153846154e06,
                    1.826923076923078e07,
                    -4.807692307692308e05,
                    8.653846153846158e06,
                    -2.403846153846154e06,
                    -9.134615384615388e06,
                    4.807692307692308e05,
                    -1.778846153846155e07,
                ],
                [
                    1.442307692307694e06,
                    -4.807692307692308e05,
                    6.250000000000003e06,
                    -2.403846153846154e06,
                    -4.567307692307696e06,
                    4.807692307692307e05,
                    -3.125000000000001e06,
                    2.403846153846154e06,
                ],
                [
                    4.807692307692308e05,
                    8.653846153846158e06,
                    -2.403846153846154e06,
                    1.826923076923078e07,
                    -4.807692307692308e05,
                    -1.778846153846155e07,
                    2.403846153846154e06,
                    -9.134615384615388e06,
                ],
                [
                    -3.125000000000001e06,
                    -2.403846153846154e06,
                    -4.567307692307696e06,
                    -4.807692307692308e05,
                    6.250000000000003e06,
                    2.403846153846154e06,
                    1.442307692307694e06,
                    4.807692307692307e05,
                ],
                [
                    -2.403846153846154e06,
                    -9.134615384615388e06,
                    4.807692307692307e05,
                    -1.778846153846155e07,
                    2.403846153846154e06,
                    1.826923076923078e07,
                    -4.807692307692308e05,
                    8.653846153846158e06,
                ],
                [
                    -4.567307692307697e06,
                    4.807692307692308e05,
                    -3.125000000000001e06,
                    2.403846153846154e06,
                    1.442307692307694e06,
                    -4.807692307692308e05,
                    6.250000000000003e06,
                    -2.403846153846154e06,
                ],
                [
                    -4.807692307692308e05,
                    -1.778846153846155e07,
                    2.403846153846154e06,
                    -9.134615384615388e06,
                    4.807692307692307e05,
                    8.653846153846158e06,
                    -2.403846153846154e06,
                    1.826923076923078e07,
                ],
            ]
        )

        np.testing.assert_array_almost_equal(
            mech2d.elem_stiffness(self.x, self.y, self.C, element.BLQuadQ4),
            expected,
            decimal=4,
        )

    def test_elem_stiffness_8x4(self):
        expected = io.loadmat(
            path.join(
                path.dirname(path.abspath(__file__)),
                "expected/mech2d_elem_stiffness_8x4.mat",
            )
        )["elem_stiff"]

        x = np.array([14.0, 16, 16, 14])
        y = np.array([1.5, 1.5, 2.0, 2.0])

        np.testing.assert_array_almost_equal(
            mech2d.elem_stiffness(x, y, self.C, element.BLQuadQ4), expected, decimal=4
        )

    def test_elem_mass_shape(self):
        np.testing.assert_array_equal(
            mech2d.elem_mass(self.x, self.y, 5.0, element.BLQuadQ4).shape, (8, 8)
        )

    def test_grid_mesh(self):
        xs = self.xs
        ys = self.ys
        expected_coords = np.array(
            [
                [
                    0.000000000000000e00,
                    4.000000000000000e00,
                    8.000000000000000e00,
                    1.200000000000000e01,
                    1.600000000000000e01,
                    0.000000000000000e00,
                    4.000000000000000e00,
                    8.000000000000000e00,
                    1.200000000000000e01,
                    1.600000000000000e01,
                    0.000000000000000e00,
                    4.000000000000000e00,
                    8.000000000000000e00,
                    1.200000000000000e01,
                    1.600000000000000e01,
                ],
                [
                    0.000000000000000e00,
                    0.000000000000000e00,
                    0.000000000000000e00,
                    0.000000000000000e00,
                    0.000000000000000e00,
                    1.000000000000000e00,
                    1.000000000000000e00,
                    1.000000000000000e00,
                    1.000000000000000e00,
                    1.000000000000000e00,
                    2.000000000000000e00,
                    2.000000000000000e00,
                    2.000000000000000e00,
                    2.000000000000000e00,
                    2.000000000000000e00,
                ],
            ]
        ).T
        expected_elem_nodes = (
            np.array(
                [
                    [1, 2, 3, 4, 6, 7, 8, 9],
                    [2, 3, 4, 5, 7, 8, 9, 10],
                    [7, 8, 9, 10, 12, 13, 14, 15],
                    [6, 7, 8, 9, 11, 12, 13, 14],
                ]
            ).T
            - 1
        )
        mesh = mech2d.grid_mesh(xs, ys)
        np.testing.assert_array_almost_equal(mesh.nodes_coords, expected_coords)
        np.testing.assert_array_equal(mesh.elems_nodes, expected_elem_nodes)

    def test_mech2d_nobigm(self):
        force = None

        def g(x, y):
            if np.isclose(x, 0.0) and np.isclose(y, 0.0):
                return [0.0, 0.0]
            elif np.isclose(x, 0.0) and np.isclose(y, self.c):
                return [0.0, None]
            elif np.isclose(y, 0.0):
                return [0.0, None]
            else:
                return [None, None]

        expected = io.loadmat(
            path.join(
                path.dirname(path.abspath(__file__)),
                "expected/mech2d_mech2d_nobigm_bigk.mat",
            )
        )["bigk"]

        sosys = mech2d.mech2d(self.xs, self.ys, self.rho, self.C, g, force, 1)

        try:
            nonzero = sosys.K.nnz
        except AttributeError:
            nonzero = len(sosys.K.nonzero()[0])

        self.assertEqual(nonzero, expected.nnz)
        try:
            toarray = sosys.K.toarray()
        except AttributeError:
            toarray = sosys.K
        np.testing.assert_array_almost_equal(toarray, expected.toarray(), decimal=4)

    def test_mech2d_nobigm_8x4(self):
        force = None

        def g(x, y):
            if np.isclose(x, 0.0) and np.isclose(y, 0.0):
                return [0.0, 0.0]
            elif np.isclose(x, 0.0) and np.isclose(y, self.c):
                return [0.0, None]
            elif np.isclose(y, 0.0):
                return [0.0, None]
            else:
                return [None, None]

        expected = io.loadmat(
            path.join(
                path.dirname(path.abspath(__file__)),
                "expected/mech2d_mech2d_nobigm_bigk_8x4.mat",
            )
        )["bigk"]

        xs = np.linspace(0, self.L, 9)
        ys = np.linspace(0, self.c, 5)
        sosys = mech2d.mech2d(xs, ys, self.rho, self.C, g, force, 1)

        try:
            toarray = sosys.K.toarray()
        except AttributeError:
            toarray = sosys.K
        np.testing.assert_array_almost_equal(toarray, expected.toarray(), decimal=4)

    def test_parabolic_traction(self):
        traction = mech2d.parabolic_traction(self.L, self.c)
        npt.assert_array_almost_equal(
            traction(0, 0.211324865405187, -1), [-0.633974596215561, 0.370813293868264]
        )
        npt.assert_array_almost_equal(
            traction(self.L, 0.211324865405187, -1), [0.0, -0.370813293868264]
        )

    def test_timevarying_traction(self):
        mesh = mech2d.grid_mesh(self.xs, self.ys)
        traction_temp = mech2d.parabolic_traction(self.L, self.c)
        parameter = mech2d.sys.PWLFunction([0, 1, 2], [0, 0, 0])
        tv_traction = mech2d.TimeVaryingTractionForce(parameter, traction_temp, mesh)
        f_nodal = tv_traction.f_nodal
        f_nodal_true = mech2d.traction_nodal_force(
            lambda x, y: traction_temp(x, y, -1), mesh
        )
        tv_traction.ys = [-1, -1, -1]
        npt.assert_array_almost_equal(f_nodal(1), f_nodal_true)
        f_nodal_true = mech2d.traction_nodal_force(
            lambda x, y: traction_temp(x, y, -2), mesh
        )
        tv_traction.ys = [-2, -2, -2]
        npt.assert_array_almost_equal(f_nodal(1), f_nodal_true)

    def test_traction_nodal(self):
        expected = np.array(
            [
                -500.000000000000e-003,
                179.687500000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -179.687500000000e-003,
                -3.00000000000000e000,
                265.625000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -265.625000000000e-003,
                -2.50000000000000e000,
                54.6875000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -54.6875000000000e-003,
            ]
        )
        mesh = mech2d.grid_mesh(self.xs, self.ys)
        P = -1
        traction = mech2d.parabolic_traction(self.L, self.c)

        f_nodal = mech2d.traction_nodal_force(lambda x, y: traction(x, y, P), mesh)
        npt.assert_array_almost_equal(f_nodal, expected)

    def test_traction_nodal_8x4(self):
        expected = np.array(
            [
                -125.000000000000e-003,
                92.7734375000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -92.7734375000000e-003,
                -750.000000000000e-003,
                173.828125000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -173.828125000000e-003,
                -1.50000000000000e000,
                138.671875000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -138.671875000000e-003,
                -2.25000000000000e000,
                80.0781250000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -80.0781250000000e-003,
                -1.37500000000000e000,
                14.6484375000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -14.6484375000000e-003,
            ]
        )

        xs = np.linspace(0, self.L, 9)
        ys = np.linspace(0, self.c, 5)
        mesh = mech2d.grid_mesh(xs, ys)
        P = -1
        traction = mech2d.parabolic_traction(self.L, self.c)

        f_nodal = mech2d.traction_nodal_force(lambda x, y: traction(x, y, P), mesh)
        npt.assert_array_almost_equal(f_nodal, expected)

    def test_traction_nodal_q9(self):
        expected = np.array(
            [
                13.8777878078145e-018,
                64.0625000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -64.0625000000000e-003,
                -1.00000000000000e000,
                231.250000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -231.250000000000e-003,
                -1.00000000000000e000,
                96.8750000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -96.8750000000000e-003,
                -3.00000000000000e000,
                106.250000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -106.250000000000e-003,
                -1.00000000000000e000,
                1.56250000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -1.56250000000000e-003,
            ]
        )

        xs = np.linspace(0, self.L, 9)
        ys = np.linspace(0, self.c, 5)
        mesh = mech2d.grid_mesh(xs, ys, q4=False)
        P = -1
        traction = mech2d.parabolic_traction(self.L, self.c)

        f_nodal = mech2d.traction_nodal_force(lambda x, y: traction(x, y, P), mesh)
        npt.assert_array_almost_equal(f_nodal, expected)

    def test_traction_nodal_q9_2x1(self):
        expected = np.array(
            [
                55.5111512312578e-018,
                137.500000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -137.500000000000e-003,
                -4.00000000000000e000,
                350.000000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -350.000000000000e-003,
                -2.00000000000000e000,
                12.5000000000000e-003,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                0.00000000000000e000,
                -12.5000000000000e-003,
            ]
        )

        xs = np.linspace(0, self.L, 5)
        ys = np.linspace(0, self.c, 3)
        mesh = mech2d.grid_mesh(xs, ys, q4=False)
        P = -1
        traction = mech2d.parabolic_traction(self.L, self.c)

        f_nodal = mech2d.traction_nodal_force(lambda x, y: traction(x, y, P), mesh)
        npt.assert_array_almost_equal(f_nodal, expected)
