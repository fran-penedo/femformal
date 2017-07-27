import logging

import numpy as np


logger = logging.getLogger(__name__)

class Element(object):
    def __init__(self, coords):
        self.coords = coords
        self.dimension = self.coords.shape[1]

    def interpolate_phys(self, values, coords):
        return self.shapes(*self.normalize(coords)).dot(np.array(values))

    def interpolate_derivatives_phys(self, values, coords):
        return self.shapes_derivatives(*self.normalize(coords)).dot(np.array(values))

    def interpolate(self, values, coords):
        return self.shapes(*coords).dot(np.array(values))

    def interpolate_derivatives(self, values, coords):
        return self.shapes_derivatives(*coords).dot(np.array(values))

    def normalize(self, coords):
        raise NotImplementedError()

    @staticmethod
    def shapes(*parameters):
        raise NotImplementedError()

    @staticmethod
    def shapes_derivatives(*parameters):
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

    def _normalize_full(self, coords):
        return self.transf_matrix.dot(coords - self.center)

    def _normalize_0d(self, coords):
        return np.array([0,0])

    def _normalize_1d(self, coords):
        return 2 * (coords - self.coords[0]) / self.ndiff - 1
