import unittest
import logging

import numpy as np
from numpy import testing as npt
from scipy.optimize import differential_evolution

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
        np.testing.assert_almost_equal(self.elem.interpolate(vs, [0, 0]), np.mean(vs))
        vs = np.array([[1,2,3,4], [4,3,2,1]]).T
        np.testing.assert_array_almost_equal(self.elem.interpolate(vs, [0, 0]),
                                             np.mean(vs, axis=0))

    def test_interpolate_phys(self):
        vs = np.array([1,2,3,4])
        np.testing.assert_equal(self.elem.interpolate_phys(vs, [0, 0.5]), np.mean(vs))
        vs = np.array([[1,2,3,4], [4,3,2,1]]).T
        np.testing.assert_array_almost_equal(self.elem.interpolate_phys(vs, [0, 0.5]),
                                             np.mean(vs, axis=0))

    def test_interpolate_strain(self):
        vs = np.array([[1,2,3,4], [4,3,2,1]]).T
        for x in np.linspace(-1.0, 1.0, 11):
            for y in np.linspace(-1.0, 2, 11):
                strain = self.elem.interpolate_strain(vs, [x, y])
                npt.assert_array_almost_equal(
                    strain[0], ((2 + 1 * ((y + 1) / 3)) - (1 + 3 * ((y + 1) / 3))) / 2)
                npt.assert_array_almost_equal(
                    strain[1], ((1 + 1 * ((x + 1) / 2)) - (4 + (-1) * ((x + 1) / 2))) / 3)

    def test_interpolate_derivatives_phys_1dof(self):
        vs = np.array([1,2,3,4])
        for x in np.linspace(-1.0, 1.0, 11):
            for y in np.linspace(-1.0, 2, 11):
                derivs = self.elem.interpolate_derivatives_phys(vs, [x, y])[0]
                npt.assert_array_almost_equal(
                    derivs[0], ((2 + 1 * ((y + 1) / 3)) - (1 + 3 * ((y + 1) / 3))) / 2)
                npt.assert_array_almost_equal(
                    derivs[1], ((4 + (-1) * ((x + 1) / 2)) - (1 + 1 * ((x + 1) / 2))) / 3)

    def test_interpolate_derivatives_phys_2dof(self):
        vs = np.array([[1,2,3,4], [4,3,2,1]]).T
        for x in np.linspace(-1.0, 1.0, 11):
            for y in np.linspace(-1.0, 2, 11):
                derivs = self.elem.interpolate_derivatives_phys(vs, [x, y])
                npt.assert_array_almost_equal(
                    derivs[0, 0], ((2 + 1 * ((y + 1) / 3)) - (1 + 3 * ((y + 1) / 3))) / 2)
                npt.assert_array_almost_equal(
                    derivs[0, 1], ((4 + (-1) * ((x + 1) / 2)) - (1 + 1 * ((x + 1) / 2))) / 3)
                npt.assert_array_almost_equal(
                    derivs[1, 0], ((3 + (-1) * ((y + 1) / 3)) - (4 + (-3) * ((y + 1) / 3))) / 2)
                npt.assert_array_almost_equal(
                    derivs[1, 1], ((1 + 1 * ((x + 1) / 2)) - (4 + (-1) * ((x + 1) / 2))) / 3)


    def test_jacobian(self):
        x = np.array([12.0,16,16,12])
        y = np.array([1.0, 1.0, 2.0, 2.0])
        pts = [-1/np.sqrt(3), 1/np.sqrt(3)]

        expected = np.array([[4.999999999999999e-01, -0.000000000000000e+00],
                             [-0.000000000000000e+00, 2.000000000000000e+00]])
        expected_det = 1.0

        for i in range(len(pts)):
            for j in range(len(pts)):
                jac_inv, jac_det = element.BLQuadQ4.jacobian(pts[i], pts[j], x, y)
                np.testing.assert_array_almost_equal(jac_inv, expected, decimal=4)
                np.testing.assert_almost_equal(jac_det, expected_det, decimal=4)

    def test_jacobian_8x4(self):
        x = np.array([14.0,16,16,14])
        y = np.array([1.5, 1.5, 2.0, 2.0])
        pts = [-1/np.sqrt(3), 1/np.sqrt(3)]

        expected_det = 250.000000000000e-003
        expected = np.array([[250.000000000000e-003, 0.00000000000000e+000],
                             [0.00000000000000e+000, 1.00000000000000e+000]]) / expected_det

        for i in range(len(pts)):
            for j in range(len(pts)):
                jac_inv, jac_det = element.BLQuadQ4.jacobian(pts[i], pts[j], x, y)
                np.testing.assert_array_almost_equal(jac_inv, expected, decimal=4)
                np.testing.assert_almost_equal(jac_det, expected_det, decimal=4)

    def test_strain_displacement(self):
        x = np.array([12,16,16,12])
        y = np.array([1.0, 1.0, 2.0, 2.0])
        pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
        expected = [[np.array([[-1.971687836487032e-01, 0.000000000000000e+00, 1.971687836487032e-01, 0.000000000000000e+00, 5.283121635129676e-02, 0.000000000000000e+00, -5.283121635129676e-02, 0.000000000000000e+00],
                               [0.000000000000000e+00, -7.886751345948131e-01, 0.000000000000000e+00, -2.113248654051871e-01, 0.000000000000000e+00, 2.113248654051871e-01, 0.000000000000000e+00, 7.886751345948131e-01],
                               [-7.886751345948131e-01, -1.971687836487032e-01, -2.113248654051871e-01, 1.971687836487032e-01, 2.113248654051871e-01, 5.283121635129676e-02, 7.886751345948131e-01, -5.283121635129676e-02]]),
                     np.array([[-5.283121635129676e-02, 0.000000000000000e+00, 5.283121635129676e-02, 0.000000000000000e+00, 1.971687836487032e-01, 0.000000000000000e+00, -1.971687836487032e-01, 0.000000000000000e+00],
                               [0.000000000000000e+00, -7.886751345948131e-01, 0.000000000000000e+00, -2.113248654051871e-01, 0.000000000000000e+00, 2.113248654051871e-01, 0.000000000000000e+00, 7.886751345948131e-01],
                               [-7.886751345948131e-01, -5.283121635129676e-02, -2.113248654051871e-01, 5.283121635129676e-02, 2.113248654051871e-01, 1.971687836487032e-01, 7.886751345948131e-01, -1.971687836487032e-01]])],
                    [np.array([[-1.971687836487032e-01, 0.000000000000000e+00, 1.971687836487032e-01, 0.000000000000000e+00, 5.283121635129676e-02, 0.000000000000000e+00, -5.283121635129676e-02, 0.000000000000000e+00],
                               [0.000000000000000e+00, -2.113248654051871e-01, 0.000000000000000e+00, -7.886751345948131e-01, 0.000000000000000e+00, 7.886751345948131e-01, 0.000000000000000e+00, 2.113248654051871e-01],
                               [-2.113248654051871e-01, -1.971687836487032e-01, -7.886751345948131e-01, 1.971687836487032e-01, 7.886751345948131e-01, 5.283121635129676e-02, 2.113248654051871e-01, -5.283121635129676e-02]]),
                     np.array([[-5.283121635129676e-02, 0.000000000000000e+00, 5.283121635129676e-02, 0.000000000000000e+00, 1.971687836487032e-01, 0.000000000000000e+00, -1.971687836487032e-01, 0.000000000000000e+00],
                               [0.000000000000000e+00, -2.113248654051871e-01, 0.000000000000000e+00, -7.886751345948131e-01, 0.000000000000000e+00, 7.886751345948131e-01, 0.000000000000000e+00, 2.113248654051871e-01],
                               [-2.113248654051871e-01, -5.283121635129676e-02, -7.886751345948131e-01, 5.283121635129676e-02, 7.886751345948131e-01, 1.971687836487032e-01, 2.113248654051871e-01, -1.971687836487032e-01]])]]

        for i in range(len(pts)):
            for j in range(len(pts)):
                strain_disp = element.BLQuadQ4.strain_displacement(pts[i], pts[j], x, y)
                np.testing.assert_array_almost_equal(
                    strain_disp, expected[i][j], decimal=4)

    def test_strain_displacement_8x4(self):
        x = np.array([14.0,16,16,14])
        y = np.array([1.5, 1.5, 2.0, 2.0])
        pts = [-1/np.sqrt(3), 1/np.sqrt(3)]

        expected = [[np.array([[-98.5843918243516e-003, 0.00000000000000e+000, 98.5843918243516e-003, 0.00000000000000e+000, 26.4156081756484e-003, 0.00000000000000e+000, -26.4156081756484e-003, 0.00000000000000e+000],
                               [0.00000000000000e+000, -394.337567297407e-003, 0.00000000000000e+000, -105.662432702594e-003, 0.00000000000000e+000, 105.662432702594e-003, 0.00000000000000e+000, 394.337567297407e-003],
                               [-394.337567297407e-003, -98.5843918243516e-003, -105.662432702594e-003, 98.5843918243516e-003, 105.662432702594e-003, 26.4156081756484e-003, 394.337567297407e-003, -26.4156081756484e-003]]),
                     np.array([[-26.4156081756484e-003, 0.00000000000000e+000, 26.4156081756484e-003, 0.00000000000000e+000, 98.5843918243516e-003, 0.00000000000000e+000, -98.5843918243516e-003, 0.00000000000000e+000 ],
                               [0.00000000000000e+000, -394.337567297407e-003, 0.00000000000000e+000, -105.662432702594e-003, 0.00000000000000e+000, 105.662432702594e-003, 0.00000000000000e+000, 394.337567297407e-003 ],
                               [-394.337567297407e-003, -26.4156081756484e-003, -105.662432702594e-003, 26.4156081756484e-003, 105.662432702594e-003, 98.5843918243516e-003, 394.337567297407e-003, -98.5843918243516e-003 ]])],
                    [np.array([[-98.5843918243516e-003, 0.00000000000000e+000, 98.5843918243516e-003, 0.00000000000000e+000, 26.4156081756484e-003, 0.00000000000000e+000, -26.4156081756484e-003, 0.00000000000000e+000],
                               [0.00000000000000e+000, -105.662432702594e-003, 0.00000000000000e+000, -394.337567297407e-003, 0.00000000000000e+000, 394.337567297407e-003, 0.00000000000000e+000, 105.662432702594e-003],
                               [-105.662432702594e-003, -98.5843918243516e-003, -394.337567297407e-003, 98.5843918243516e-003, 394.337567297407e-003, 26.4156081756484e-003, 105.662432702594e-003, -26.4156081756484e-003]]) ,
                     np.array([[-26.4156081756484e-003, 0.00000000000000e+000, 26.4156081756484e-003, 0.00000000000000e+000, 98.5843918243516e-003, 0.00000000000000e+000, -98.5843918243516e-003, 0.00000000000000e+000],
                               [0.00000000000000e+000, -105.662432702594e-003, 0.00000000000000e+000, -394.337567297407e-003, 0.00000000000000e+000, 394.337567297407e-003, 0.00000000000000e+000, 105.662432702594e-003],
                               [-105.662432702594e-003, -26.4156081756484e-003, -394.337567297407e-003, 26.4156081756484e-003, 394.337567297407e-003, 98.5843918243516e-003, 105.662432702594e-003, -98.5843918243516e-003]])]]

        for i in range(len(pts)):
            for j in range(len(pts)):
                _, jac_det = element.BLQuadQ4.jacobian(pts[i], pts[j], x, y)
                strain_disp = jac_det * element.BLQuadQ4.strain_displacement(pts[i], pts[j], x, y)
                np.testing.assert_array_almost_equal(
                    strain_disp, expected[i][j], decimal=4)

    def test_max_diff(self):
        vs = np.array([0, 1, 2, 3]).reshape(4,1)
        npt.assert_equal(self.elem.max_diff(vs, axis=0), 2)
        npt.assert_equal(self.elem.max_diff(vs, axis=1), 3)
        vs = np.array([[0, 1, 2, 3], [0, 1, 2, 3]]).T
        npt.assert_array_equal(self.elem.max_diff(vs, axis=0), [2, 2])
        npt.assert_array_equal(self.elem.max_diff(vs, axis=1), [3, 3])


    def test_chebyshev_radius(self):
        npt.assert_almost_equal(self.elem.chebyshev_radius(), np.sqrt(13) / 2.0)

    def test_chebyshev_center(self):
        npt.assert_almost_equal(self.elem.chebyshev_center(), [0, 0.5])


class TestQuadQuadQ9(unittest.TestCase):
    def setUp(self):
        self.elem = element.QuadQuadQ9(np.array([
            [-1.0, -1], [1, -1], [1, 2], [-1, 2],
            [0, -1], [1, 0.5], [0, 2], [-1, 0.5], [0, 0.5]]))
        self.elem1Dh = element.QuadQuadQ9(np.array([
            [-1.0, -1], [1, -1], [1, -1], [-1, -1],
            [0, -1], [1, -1], [0, -1], [-1, -1], [0, -1]]))
        self.elem1Dv = element.QuadQuadQ9(np.array([
            [-1.0, -1], [-1, -1], [-1, 2], [-1, 2],
            [-1, -1], [-1, .5], [-1, 2], [-1, .5], [-1, .5]]))
        self.elem0D = element.QuadQuadQ9(np.array([[-1, -1] for i in range(9)]))

    def test_shapes(self):
        pts = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1],
                        [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
        expected = np.identity(pts.shape[0])
        for i in range(pts.shape[0]):
            np.testing.assert_array_equal(self.elem.shapes(*pts[i, :]), expected[i])

    def test_shapes_derivatives(self):
        pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
        expected = [[np.array([[-0.4906, -0.0352, 0.0094, 0.1314, 0.5258, -0.0516, -0.1409, -0.7182, 0.7698],
                               [-0.4906, 0.1314, 0.0094, -0.0352, -0.7182, -0.1409, -0.0516, 0.5258, 0.7698]]),
                     np.array([[0.1314, 0.0094, -0.0352, -0.4906, -0.1409, -0.0516, 0.5258, -0.7182, 0.7698],
                               [0.0352, -0.0094, -0.1314, 0.4906, 0.0516, 0.1409, 0.7182, -0.5258, -0.7698]])],
                    [np.array([[0.0352, 0.4906, -0.1314, -0.0094, -0.5258, 0.7182, 0.1409, 0.0516, -0.7698],
                               [0.1314, -0.4906, -0.0352, 0.0094, -0.7182, 0.5258, -0.0516, -0.1409, 0.7698]]),
                     np.array([[-0.0094, -0.1314, 0.4906, 0.0352, 0.1409, 0.7182, -0.5258, 0.0516, -0.7698],
                               [-0.0094, 0.0352, 0.4906, -0.1314, 0.0516, -0.5258, 0.7182, 0.1409, -0.7698]])]]
        for i in range(len(pts)):
            for j in range(len(pts)):
                np.testing.assert_array_almost_equal(
                    self.elem.shapes_derivatives(pts[i], pts[j]),
                    expected[i][j], decimal=4)


    def test_normalize(self):
        test = self.elem.coords
        expected = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1],
                        [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
        for t, e in zip(test, expected):
            np.testing.assert_array_equal(self.elem.normalize(t), e)

    def test_covering(self):
        expected_0D = [(np.array([-1, -1]), 0)]
        expected_1Dh = [(np.array([-.5, -1]), .5), (np.array([.5, -1]), .5)]
        expected_1Dv = [(np.array([-1, -.25]), .75), (np.array([-1, 1.25]), .75)]
        h = np.sqrt(1 + 1.5*1.5) / 2
        expected_full = [(np.array([-.5, -.25]), h), (np.array([-.5, 1.25]), h),
            (np.array([.5, -.25]), h), (np.array([.5, 1.25]), h)]

        for a, b in zip(self.elem0D._2elem_covering(), expected_0D):
            npt.assert_array_almost_equal(a[0], b[0])
            npt.assert_array_almost_equal(a[1], b[1])

        for a, b in zip(self.elem1Dh._2elem_covering(), expected_1Dh):
            npt.assert_array_almost_equal(a[0], b[0])
            npt.assert_array_almost_equal(a[1], b[1])

        for a, b in zip(self.elem1Dv._2elem_covering(), expected_1Dv):
            npt.assert_array_almost_equal(a[0], b[0])
            npt.assert_array_almost_equal(a[1], b[1])

        for a, b in zip(self.elem._2elem_covering(), expected_full):
            npt.assert_array_almost_equal(a[0], b[0])
            npt.assert_array_almost_equal(a[1], b[1])


    def test_max_partial_derivs(self):
        np.random.seed(0)
        vs = np.random.random((9, 2))
        derivs = self.elem.max_partial_derivs(vs)
        bounds = [(-1, 1), (-1, 1)]

        f00 = lambda x: -np.abs(self.elem.interpolate_derivatives(vs, x)[0, 0])
        res = differential_evolution(f00, bounds, tol=0.0001)
        m00 = -res.fun
        logger.debug(res.x)
        npt.assert_almost_equal(derivs[0, 0], m00, decimal=3)

        f10 = lambda x: -np.abs(self.elem.interpolate_derivatives(vs, x)[1, 0])
        res = differential_evolution(f10, bounds, tol=0.0001)
        m10 = -res.fun
        logger.debug(res.x)
        npt.assert_almost_equal(derivs[1, 0], m10, decimal=3)

        f01 = lambda x: -np.abs(self.elem.interpolate_derivatives(vs, x)[0, 1])
        res = differential_evolution(f01, bounds, tol=0.0001)
        m01 = -res.fun
        logger.debug(res.x)
        npt.assert_almost_equal(derivs[0, 1], m01, decimal=3)

        f11 = lambda x: -np.abs(self.elem.interpolate_derivatives(vs, x)[1, 1])
        res = differential_evolution(f11, bounds, tol=0.0001)
        m11 = -res.fun
        logger.debug(res.x)
        npt.assert_almost_equal(derivs[1, 1], m11, decimal=3)

