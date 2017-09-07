import unittest
import logging

import numpy as np
from numpy import testing as npt

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
        x = np.array([12,16,16,12])
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
