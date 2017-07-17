import logging

import numpy as np


logger = logging.getLogger(__name__)

class Element(object):
    def __init__(self, coords):
        self.coords = coords

    def interpolate(self, values, coords):
        return self.shapes(*coords).dot(values)

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
