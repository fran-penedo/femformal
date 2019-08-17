"""
FEM element data structures and definitions
"""
from __future__ import division, absolute_import, print_function

import logging
import abc

import numpy as np

from .. import util


logger = logging.getLogger(__name__)


class Element(object):
    """Abstract element

    Parameters
    ----------
    coords : array, shape (nnodes, 2)
        Coordinates of the nodes
    shape_order : int, optional
        Maximum order of the polynomials used as shape functions

    Attributes
    ----------
    coords : array
    dimension : int
        Dimension of the element
    shape_order : int

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, coords, shape_order=0):
        self.coords = coords
        self.dimension = self.coords.shape[1]
        self.shape_order = shape_order


class Element2DOF(Element):
    """Abstract 2D element with 2 DOFs

    Parameters
    ----------
    coords : array, shape (nnodes, 2)
        Coordinates of the nodes
    shape_order : int, optional
        Maximum order of the polynomials used as shape functions

    Attributes
    ----------
    coords : array
    dimension : int
        Dimension of the element
    shape_order : int

    """

    def __init__(self, coords, shape_order=0):
        Element.__init__(self, coords, shape_order)

    def interpolate_phys(self, values, coords):
        """Interpolates values at the nodes at some physical coordinates

        Parameters
        ----------
        values : array, shape (nnodes, 2)
            Values asigned to each node
        coords : array, shape (1, 2)
            Physical coordinates

        Returns
        -------
        array, shape (1, 2)
            Interpolation of `values` at `coords`

        """
        return self.shapes(*self.normalize(coords)).dot(np.array(values))

    def interpolate_strain(self, values, coords):
        """Interpolates the strain at some coordinates

        Parameters
        ----------
        values : array, shape (nnodes, 2)
            Values asigned to each node
        coords : array, shape (1, 2)
            Physical coordinates

        Returns
        -------
        array, shape (3, 1)
            Interpolation of the strain at `coords`

        """
        a, b = self.normalize(coords)
        x, y = zip(*self.coords)
        values = np.array(values)
        if len(values.shape) == 3:
            v = values.reshape(values.shape[0] * values.shape[1], values.shape[2])
        else:
            v = values.reshape(np.prod(values.shape))
        return self.strain_displacement(a, b, x, y).dot(v)

    def interpolate(self, values, coords):
        """Interpolates values at the nodes at some coordinates

        Parameters
        ----------
        values : array, shape (nnodes, 2)
            Values asigned to each node
        coords : array, shape (1, 2)
            Psi-Eta coordinates

        Returns
        -------
        array, shape (1, 2)
            Interpolation of `values` at `coords`

        """
        return self.shapes(*coords).dot(np.array(values))

    def interpolate_derivatives(self, values, coords):
        """Interpolates derivatives of values at the nodes at some coordinates

        Parameters
        ----------
        values : array, shape (nnodes, 2)
            Values asigned to each node
        coords : array, shape (1, 2)
            Psi-Eta coordinates

        Returns
        -------
        array, shape (2, 2)
            Interpolation of the derivatives of `values` at `coords`

        """
        a, b = coords
        x, y = zip(*self.coords)
        values = np.array(values)
        jac_inv, jac_det = self.jacobian(a, b, x, y)
        dshapes = jac_inv.dot(self.shapes_derivatives(a, b))
        return dshapes.dot(values)

    def interpolate_derivatives_phys(self, values, coords):
        """Deprecated

        Designed for scalar fields or vector fields with same dimension as
        domain

        """
        a, b = self.normalize(coords)
        x, y = zip(*self.coords)
        str_disp = self.strain_displacement(a, b, x, y)
        values = np.array(values)
        if np.prod(values.shape) != str_disp.shape[-1]:
            dof = 1 if len(values.shape) == 1 else values.shape[-1]
            values = np.c_[
                values,
                np.zeros(
                    (values.shape[0], str_disp.shape[-1] // np.prod(values.shape) - dof)
                ),
            ]

        ret = np.zeros((values.shape[-1], values.shape[-1]))
        for p in util.cycle(values.shape[-1]):
            p = list(p)
            v = values.T[np.ix_(p)].T
            v = v.reshape(np.prod(v.shape))
            derivs = str_disp.dot(v)[0 : values.shape[-1]]
            ret[tuple([p, range(values.shape[-1])])] = derivs

        return ret

    @abc.abstractmethod
    def normalize(self, coords):
        """Transforms physical coordinates into Psi-Eta coordinates

        Parameters
        ----------
        coords : array, shape (1, 2)
            The physical coordinates

        Returns
        -------
        array, shape (1, 2)
            The corresponding Psi-Eta coordinates
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def shapes(*parameters):
        """Shape functions of the element

        Parameters
        ----------
        parameters
            Arguments of the shape functions

        Returns
        -------
        array, shape (nnodes)
            Shape functions evaluated at the `parameters`

        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def shapes_derivatives(*parameters):
        """Derivatives of the shape functions of the element

        Parameters
        ----------
        parameters
            Arguments of the shape functions

        Returns
        -------
        array, shape (len(parameters), nnodes)
            Derivatives of the shape functions evaluated at the `parameters`

        """
        raise NotImplementedError()

    @classmethod
    def strain_displacement(cls, *parameters):
        """Strain-displacement matrix of the element

        Parameters
        ----------
        parameters
            Arguments of the shape functions

        Returns
        -------
        array, shape (3, 2 * nnodes)
            Strain-displacent matrix evaluated at `parameters`

        """
        a, b, x, y = parameters
        jac_inv, jac_det = cls.jacobian(a, b, x, y)
        dshapes_phy = jac_inv.dot(cls.shapes_derivatives(a, b))
        n = 2 * dshapes_phy.shape[1]
        str_disp = np.zeros((3, n))
        str_disp[0, 0:n:2] = dshapes_phy[0]
        str_disp[1, 1:n:2] = dshapes_phy[1]
        str_disp[2, 0:n:2] = dshapes_phy[1]
        str_disp[2, 1:n:2] = dshapes_phy[0]

        return str_disp

    @classmethod
    def jacobian(cls, a, b, x, y):
        """Jacobian of the change of coordinates at some point

        Parameters
        ----------
        a, b : float
            Psi-Eta coordinates
        x, y : array, shape (1, nnodes)
            Coordinates of the nodes

        Returns
        -------
        jac_inv : array, shape(2, 2)
            Inverse of the Jacobian
        jac_det : float
            Determinant of the Jacobian

        """
        jac = cls.shapes_derivatives(a, b).dot(np.vstack([x, y]).T)
        jac_det = np.linalg.det(jac)
        jac_inv = np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / jac_det
        return jac_inv, jac_det

    @abc.abstractmethod
    def max_diff(self, values, axis):
        raise NotImplementedError()


class BLQuadQ4(Element2DOF):
    """Bilinear quadrilateral element with 4 nodes

    Nodes are numbered in the following way, starting in index 0:

        3 - 2
        |   |
        0 - 1

    Parameters
    ----------
    coords : array, shape (4, 2)
        Coordinates of the nodes

    Attributes
    ----------
    center : array, shape (1, 2)
        Center of the element

    """

    def __init__(self, coords):
        if coords.shape != (4, 2):
            raise ValueError("Q4 element coordinates must be 4x2")
        Element2DOF.__init__(self, coords)
        self.center = np.mean(coords, axis=0)
        if all([np.all([np.isclose(p, self.center) for p in coords])]):
            self.normalize = self._normalize_0d
        else:
            try:
                self.transf_matrix = self._transf_matrix(coords)
                self.normalize = self._normalize_full
            except ValueError:
                self.diff = self.coords[2] - self.coords[0]
                self.ndiff = np.linalg.norm(self.diff)
                self.normalize = self._normalize_1d

    @staticmethod
    def _transf_matrix(coords):
        x, y = coords.T
        det = 2 * ((-x[0] + x[2]) * (-y[1] + y[3]) + (x[1] - x[3]) * (-y[0] + y[2]))
        if not np.isclose(det, 0):
            return (
                4
                / det
                * np.array(
                    [
                        [-y[0] - y[1] + y[2] + y[3], x[0] + x[1] - x[2] - x[3]],
                        [y[0] - y[1] - y[2] + y[3], -x[0] + x[1] + x[2] - x[3]],
                    ]
                )
            )
        else:
            raise ValueError("Cannot compute change of basis for degenerate element")

    @staticmethod
    def shapes(*parameters):
        a, b = parameters
        return 0.25 * np.array(
            [(1 - a) * (1 - b), (1 + a) * (1 - b), (1 + a) * (1 + b), (1 - a) * (1 + b)]
        )

    @staticmethod
    def shapes_derivatives(*parameters):
        a, b = parameters
        return 0.25 * np.array(
            [
                [-(1 - b), (1 - b), (1 + b), -(1 + b)],
                [-(1 - a), -(1 + a), (1 + a), (1 - a)],
            ]
        )

    def _normalize_full(self, coords):
        return self.transf_matrix.dot(coords - self.center)

    def _normalize_0d(self, coords):
        return np.array([0, 0])

    def _normalize_1d(self, coords):
        return 2 * (coords - self.coords[0]) / self.ndiff - 1

    def normalize(self, coords):
        pass

    def max_diff(self, values, axis):
        values = np.array(values)

        def take(i):
            return values.take(i, axis=-2)

        if axis == 0:
            return np.max(
                np.abs(
                    [
                        take(0) - take(1),
                        take(3) - take(2),
                        take(0) - take(2),
                        take(3) - take(1),
                    ]
                ),
                axis=0,
            )
        elif axis == 1:
            return np.max(
                np.abs(
                    [
                        take(0) - take(3),
                        take(1) - take(2),
                        take(0) - take(2),
                        take(1) - take(3),
                    ]
                ),
                axis=0,
            )
        else:
            raise ValueError("Axis must be 0 or 1. Given: {}".format(axis))

    def chebyshev_radius(self):
        return np.linalg.norm(self.coords[0] - self.coords[2]) / 2.0

    def chebyshev_center(self):
        return self.interpolate(self.coords, [0, 0])

    def covering(self):
        return [(self.chebyshev_center(), self.chebyshev_radius)]

    def ishorizontal(self):
        return np.all(np.isclose(self.coords[0], self.coords[2]))

    def isvertical(self):
        return np.all(np.isclose(self.coords[0], self.coords[1]))


class QuadQuadQ9(Element2DOF):
    """Quadratic quadrilateral element with 9 nodes

    Nodes are numbered in the following way, starting in index 0:

        3 - 6 - 2
        |   |   |
        7 - 8 - 5
        |   |   |
        0 - 4 - 1

    Parameters
    ----------
    coords : array, shape (9, 2)
        Coordinates of the nodes

    Attributes
    ----------
    center : array, shape (1, 2)
        Center of the element

    """

    def __init__(self, coords):
        if coords.shape != (9, 2):
            raise ValueError("Q9 element coordinates must be 9x2")
        Element2DOF.__init__(self, coords, shape_order=2)
        self.center = coords[-1]
        if all([np.all([np.isclose(p, self.center) for p in coords[:4]])]):
            self.normalize = self._normalize_0d
        else:
            try:
                self.transf_matrix = self._transf_matrix(coords)
                self.normalize = self._normalize_full
            except ValueError:
                self.diff = self.coords[2] - self.coords[0]
                self.ndiff = np.linalg.norm(self.diff)
                self.normalize = self._normalize_1d

    @staticmethod
    def _transf_matrix(coords):
        # FIXME I'm not sure this applies as is for Q9 from Q4
        # FIXME It definitely does not
        x, y = coords.T
        det = 2 * ((-x[0] + x[2]) * (-y[1] + y[3]) + (x[1] - x[3]) * (-y[0] + y[2]))
        if not np.isclose(det, 0):
            return (
                4
                / det
                * np.array(
                    [
                        [-y[0] - y[1] + y[2] + y[3], x[0] + x[1] - x[2] - x[3]],
                        [y[0] - y[1] - y[2] + y[3], -x[0] + x[1] + x[2] - x[3]],
                    ]
                )
            )
        else:
            raise ValueError("Cannot compute change of basis for degenerate element")

    @staticmethod
    def shapes(*parameters):
        a, b = parameters
        aa = a * a
        bb = b * b
        return 0.25 * np.array(
            [
                (aa - a) * (bb - b),
                (aa + a) * (bb - b),
                (aa + a) * (bb + b),
                (aa - a) * (bb + b),
                2 * (1 - aa) * (bb - b),
                2 * (aa + a) * (1 - bb),
                2 * (1 - aa) * (bb + b),
                2 * (aa - a) * (1 - bb),
                4 * (1 - aa) * (1 - bb),
            ]
        )

    @staticmethod
    def shapes_derivatives(*parameters):
        a, b = parameters
        aa = a * a
        bb = b * b
        return 0.25 * np.array(
            [
                [
                    (2 * a - 1) * (bb - b),
                    (2 * a + 1) * (bb - b),
                    (2 * a + 1) * (bb + b),
                    (2 * a - 1) * (bb + b),
                    2 * (-2 * a) * (bb - b),
                    2 * (2 * a + 1) * (1 - bb),
                    2 * (-2 * a) * (bb + b),
                    2 * (2 * a - 1) * (1 - bb),
                    4 * (-2 * a) * (1 - bb),
                ],
                [
                    (aa - a) * (2 * b - 1),
                    (aa + a) * (2 * b - 1),
                    (aa + a) * (2 * b + 1),
                    (aa - a) * (2 * b + 1),
                    2 * (1 - aa) * (2 * b - 1),
                    2 * (aa + a) * (-2 * b),
                    2 * (1 - aa) * (2 * b + 1),
                    2 * (aa - a) * (-2 * b),
                    4 * (1 - aa) * (-2 * b),
                ],
            ]
        )

    @staticmethod
    def shapes_second_derivatives(*parameters):
        a, b = parameters
        aa = a * a
        bb = b * b
        m = self._cross_deriv_arrays()
        return np.array(
            [
                m[0, 1]
                * (
                    bb * m[1, 1] * 0.5
                    + b * m[1, 0]
                    + np.array([0, 0, 0, 0, 0, 1, 0, 1, 1])
                ),
                (m[0, 1] * a + m[0, 0]) * (m[1, 1] * b + m[1, 0]),
                (m[0, 1] * a + m[0, 0]) * (m[1, 1] * b + m[1, 0]),
                m[1, 1]
                * (
                    aa * m[0, 1] * 0.5
                    + a * m[0, 0]
                    + np.array([0, 0, 0, 0, 1, 0, 1, 0, 1])
                ),
            ]
        )

    def _normalize_full(self, coords):
        return self.transf_matrix.dot(coords - self.center)

    def _normalize_0d(self, coords):
        return np.array([0, 0])

    def _normalize_1d(self, coords):
        return 2 * (coords - self.coords[0]) / self.ndiff - 1

    def normalize(self, coords):
        pass

    def max_diff(self, values, axis):
        # FIXME implement me
        raise NotImplementedError()
        values = np.array(values)

        def take(i):
            return values.take(i, axis=-2)

        if axis == 0:
            return np.max(
                np.abs(
                    [
                        take(0) - take(1),
                        take(3) - take(2),
                        take(0) - take(2),
                        take(3) - take(1),
                    ]
                ),
                axis=0,
            )
        elif axis == 1:
            return np.max(
                np.abs(
                    [
                        take(0) - take(3),
                        take(1) - take(2),
                        take(0) - take(2),
                        take(1) - take(3),
                    ]
                ),
                axis=0,
            )
        else:
            raise ValueError("Axis must be 0 or 1. Given: {}".format(axis))

    def max_partial_derivs(self, values):
        corners = np.array([[a, b] for a in [-1, 1] for b in [-1, 1]])
        int_opt_coords = self._int_opt_coords(values.T)
        border_opt_coords = self._border_opt_coords(values.T)
        opt_coords = np.vstack([corners, int_opt_coords, border_opt_coords])
        return np.max(
            [
                np.abs(self.interpolate_derivatives(values, coords))
                for coords in opt_coords
            ],
            axis=0,
        )

    def max_second_derivs(self, values):
        corners = np.array([[a, b] for a in [-1, 1] for b in [-1, 1]])
        int_opt_coords = self._int_opt_second_coords(values.T)
        opt_coords = np.vstack([corners, int_opt_second_coords])
        # return np.max([np.abs(self.interpolate_derivatives(values, coords))
        #                for coords in opt_coords], axis=0)

    @staticmethod
    def _cross_deriv_arrays():
        return 0.25 * np.array(
            [
                [-1, 1, 1, -1, 0, 2, 0, -2, 0],  # a,0
                [2, 2, 2, 2, -4, 4, -4, 4, -8],  # a,1
                [-1, -1, 1, 1, -1, 0, 1, 0, 0],  # b,0
                [2, 2, 2, 2, 2, -2, 2, -2, -2],
            ]
        )  # b,1

    def _int_opt_coords(self, values):
        opt_coords = []
        m = self._cross_deriv_arrays()
        for v in values:
            # w.r.t a
            A = v[0] + v[1] - 2 * v[4]
            B = v[2] + v[3] - 2 * v[6]
            C = v[5] + v[7] - 2 * v[8]
            try:
                b1 = list(_q9_soeq_sol(A, B, C))
                a1 = [_q9_foeq_sol(v, m[0], m[1], m[3] * b + m[2]) for b in b1]
            except ValueError as e:
                # logger.exception(e)
                b1 = []
                a1 = []

            # w.r.t b
            A = v[0] + v[3] - 2 * v[7]
            B = v[1] + v[2] - 2 * v[5]
            C = v[4] + v[6] - 2 * v[8]
            try:
                a2 = list(_q9_soeq_sol(A, B, C))
                b2 = [_q9_foeq_sol(v, m[2], m[3], m[1] * a + m[0]) for a in a2]
            except ValueError as e:
                # logger.exception(e)
                b2 = []
                a2 = []

            b = b1 + b2
            a = a1 + a2
            pairs = np.array([a, b]).T
            opt_coords.extend(
                [ab for ab in pairs if np.all(ab >= -1) and np.all(ab <= 1)]
            )

        ret = np.array(opt_coords)
        if len(ret) == 0:
            ret = ret.reshape((0, 2))
        return ret

    def _int_opt_second_coords(self, values):
        opt_coords = []
        m = self._cross_deriv_arrays()
        for v in values:
            a = _q9_foeq_sol(v, m[0, 1], m[0, 0], m[1, 0])
            b = _q9_foeq_sol(v, m[1, 1], m[1, 0], m[0, 0])
            opt_coords.append([a, b])
        return opt_coords

    def _border_opt_coords(self, values):
        opt_coords = []
        m = self._cross_deriv_arrays()
        for v in values:
            # w.r.t b
            b1 = [-1, 1]
            a1 = [_q9_foeq_sol(v, m[0], m[1], m[3] * b + m[2]) for b in b1]

            # w.r.t a
            a2 = [-1, 1]
            b2 = [_q9_foeq_sol(v, m[2], m[3], m[1] * a + m[0]) for a in a2]

            b = b1 + b2
            a = a1 + a2
            pairs = np.array([a, b]).T
            opt_coords.extend(
                [ab for ab in pairs if np.all(ab >= -1) and np.all(ab <= 1)]
            )

        ret = np.array(opt_coords)
        if len(ret) == 0:
            ret = ret.reshape((0, 2))
        return ret

    def chebyshev_radius(self):
        return np.linalg.norm(self.coords[0] - self.coords[2]) / 2.0

    def chebyshev_center(self):
        return self.center

    def _dim(self):
        if np.all(np.isclose(self.coords, self.coords[0])):
            return 0
        elif np.any(np.isclose(self.coords[2] - self.coords[0], 0)):
            return 1
        else:
            return 2

    def _2elem_covering(self):
        dim = self._dim()

        if dim == 0:
            return [(self.coords[0], 0)]
        elif dim == 1:
            pts = np.array([[-0.5, -0.5], [0.5, 0.5]])
            h = self.chebyshev_radius() / 2
            return [(self.interpolate(self.coords, pt), h) for pt in pts]
        else:
            pts = [-0.5, 0.5]
            h = self.chebyshev_radius() / 2
            return [
                (self.interpolate(self.coords, np.array([i, j])), h)
                for i in pts
                for j in pts
            ]

    def _4elem_covering(self):
        dim = self._dim()

        if dim == 0:
            return [(self.coords[0], 0)]
        elif dim == 1:
            pts = np.array([[-0.75, -0.75], [-0.25, -0.25], [0.25, 0.25], [0.75, 0.75]])
            h = self.chebyshev_radius() / 4
            return [(self.interpolate(self.coords, pt), h) for pt in pts]
        else:
            pts = [-0.75, -0.25, 0.25, 0.75]
            h = self.chebyshev_radius() / 4
            return [
                (self.interpolate(self.coords, np.array([i, j])), h)
                for i in pts
                for j in pts
            ]

    def covering(self):
        return self._4elem_covering()

    def ishorizontal(self):
        return np.all(np.isclose(self.coords[0], self.coords[-1]))

    def isvertical(self):
        return np.all(np.isclose(self.coords[0], self.coords[1]))


def _q9_soeq_sol(a, b, c):
    A = a + b - 2 * c
    B = b - a
    C = 2 * c
    if np.isclose(A, 0):
        if not np.isclose(B, 0):
            return (-C / B,)
        else:
            raise ValueError("A == B == 0")
    else:
        discr = B ** 2 - 4 * A * C
        if discr >= 0:
            return (-B + np.sqrt(discr)) / (2 * A), (-B - np.sqrt(discr)) / (2 * A)
        else:
            raise ValueError("Negative discriminant")


def _q9_foeq_sol(d, m0, m1, n):
    div = (m1 * n).dot(d)
    if not np.isclose(div, 0):
        return -(m0 * n).dot(d) / div
    else:
        return 10000
