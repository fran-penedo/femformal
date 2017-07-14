import logging

import numpy as np


logger = logging.getLogger(__name__)

class Mesh(object):
    def __init__(self, nodes_coords):
        self.nodes_coords = nodes_coords

    @property
    def nnodes(self):
        return self.nodes_coords.shape[0]


class GridQ4(Mesh):
    def __init__(self, nodes_coords, shape):
        Mesh.__init__(self, nodes_coords)
        shape = np.array(shape)
        if np.prod(shape) != self.nnodes:
            raise ValueError(
                "Shape {} not compatible with number of nodes {}".format(
                    shape, self.nnodes))
        self.shape = shape
        self.nelems = np.prod(self.shape - 1)
        self.elems_nodes = np.array([GridQ4._elem_nodes(e, self.shape[0] - 1)
                                for e in range(self.nelems)])

    def elem_nodes(self, elem, dim=2):
        if dim == 0:
            return elem
        elif dim == 1:
            pass
        elif dim == 2:
            return self.elems_nodes[elem]

    @staticmethod
    def _elem_nodes(e, num_elems_x):
        x0 = e + e / num_elems_x
        return np.array([x0, x0 + 1, x0 + num_elems_x + 2, x0 + num_elems_x + 1])



def find_elem_with_vertex(vnode, position, elems_nodes):
    try:
        return next(e for e, nodes in enumerate(elems_nodes)
                    if nodes[position] == vnode)
    except StopIteration:
        raise ValueError("No element with node {} in position {}".format(vnode, position))

def find_node(node, nodes_coords):
    try:
        return next(n for n, coords in enumerate(nodes_coords)
                    if np.all(np.isclose(node, coords)))
    except StopIteration:
        raise ValueError("No node with coordinates {}".format(node))

def interpolate(shapes, values, coords):
    return shapes(*coords).dot(values)

