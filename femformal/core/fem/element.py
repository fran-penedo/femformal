import logging

import numpy as np

from .. import util


logger = logging.getLogger(__name__)

class Element(object):
    def __init__(self, coords):
        self.coords = coords
        self.dimension = self.coords.shape[1]

    def interpolate_phys(self, values, coords):
        return self.shapes(*self.normalize(coords)).dot(np.array(values))

    def interpolate_strain(self, values, coords):
        a, b = self.normalize(coords)
        x, y = zip(*self.coords)
        values = np.array(values)
        v = values.reshape(np.prod(values.shape))
        return self.strain_displacement(a, b, x, y).dot(v)

    def interpolate(self, values, coords):
        return self.shapes(*coords).dot(np.array(values))

    def interpolate_derivatives_phys(self, values, coords):
        """Designed for scalar fields or vector fields with same dimension as
        domain"""

        a, b = self.normalize(coords)
        x, y = zip(*self.coords)
        str_disp = self.strain_displacement(a, b, x, y)
        values = np.array(values)
        if np.prod(values.shape) != str_disp.shape[-1]:
            dof = 1 if len(values.shape) == 1 else values.shape[-1]
            values = np.c_[
                values,
                np.zeros((values.shape[0],
                          str_disp.shape[-1] / np.prod(values.shape) - dof))]

        ret = np.zeros((values.shape[-1], values.shape[-1]))
        for p in util.cycle(values.shape[-1]):
            p = list(p)
            v = values.T[np.ix_(p)].T
            v = v.reshape(np.prod(v.shape))
            derivs = str_disp.dot(v)[0:values.shape[-1]]
            ret[[p, range(values.shape[-1])]] = derivs

        return ret


    def normalize(self, coords):
        raise NotImplementedError()

    @staticmethod
    def shapes(*parameters):
        raise NotImplementedError()

    @staticmethod
    def shapes_derivatives(*parameters):
        raise NotImplementedError()

    @staticmethod
    def strain_displacement(*parameters):
        raise NotImplementedError()

    def max_diff(self, values, axis):
        raise NotImplementedError()


class BLQuadQ4(Element):
    def __init__(self, coords):
        if coords.shape != (4, 2):
            raise ValueError("Q4 element coordinates must be 4x2")
        Element.__init__(self, coords)
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
            return 4 / det * np.array([
                [-y[0] - y[1] + y[2] + y[3], x[0] + x[1] - x[2] - x[3]],
                [y[0] - y[1] - y[2] + y[3],  -x[0] + x[1] + x[2] - x[3]]
            ])
        else:
            raise ValueError("Cannot compute change of basis for degenerate element")

    @staticmethod
    def shapes(*parameters):
        a, b = parameters
        return 0.25 * np.array([(1 - a)*(1 - b), (1 + a) * (1 - b),
                                (1 + a) * (1 + b), (1 - a) * (1 + b)])

    @staticmethod
    def shapes_derivatives(*parameters):
        a, b = parameters
        return 0.25 * np.array([[- (1 - b), (1 - b), (1 + b), - (1 + b)],
                                [- (1 - a), - (1 + a), (1 + a), (1 - a)]])

    @staticmethod
    def strain_displacement(*parameters):
        a, b, x, y = parameters
        jac_inv, jac_det = BLQuadQ4.jacobian(a, b, x, y)
        dshapes_phy = jac_inv.dot(BLQuadQ4.shapes_derivatives(a, b))
        str_disp = np.zeros((3,8))
        str_disp[0, 0:8:2] = dshapes_phy[0]
        str_disp[1, 1:8:2] = dshapes_phy[1]
        str_disp[2, 0:8:2] = dshapes_phy[1]
        str_disp[2, 1:8:2] = dshapes_phy[0]

        return str_disp

    @staticmethod
    def jacobian(a, b, x, y):
        jac = BLQuadQ4.shapes_derivatives(a,b).dot(np.vstack([x,y]).T)
        jac_det = np.linalg.det(jac)
        jac_inv = np.array([[jac[1,1], -jac[0,1]], [-jac[1,0], jac[0,0]]]) / jac_det
        return jac_inv, jac_det

    def _normalize_full(self, coords):
        return self.transf_matrix.dot(coords - self.center)

    def _normalize_0d(self, coords):
        return np.array([0,0])

    def _normalize_1d(self, coords):
        return 2 * (coords - self.coords[0]) / self.ndiff - 1

    def max_diff(self, values, axis):
        values = np.array(values)
        def take(i):
            return values.take(i, axis=-2)

        if axis == 0:
            return np.max(np.abs([take(0) - take(1), take(3) - take(2),
                                  take(0) - take(2), take(3) - take(1)]),
                          axis=0)
        elif axis == 1:
            return np.max(np.abs([take(0) - take(3), take(1) - take(2),
                                  take(0) - take(2), take(1) - take(3)]),
                          axis=0)
        else:
            raise ValueError("Axis must be 0 or 1. Given: {}".format(axis))

    def chebyshev_radius(self):
        return np.linalg.norm(self.coords[0] - self.coords[2]) / 2.0

    def chebyshev_center(self):
        return self.interpolate(self.coords, [0, 0])
