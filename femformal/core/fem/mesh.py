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
        self.elems1d_nodes = np.array(
            [GridQ4._elem_nodes(e, self.shape)
             for e in range(2 * self.nnodes - np.sum(self.shape))])

    def elem_nodes(self, elem, dim=2):
        if dim == 0:
            return [elem for i in range(4)]
        elif dim == 1:
            return self.elems1d_nodes[elem]
        elif dim == 2:
            return self.elems_nodes[elem]

    def elem_coords(self, elem, dim=2):
        return np.array([self.nodes_coords[n] for n in self.elem_nodes(elem, dim)])


    @staticmethod
    def _elem1d_nodes(e, shape):
        nelems1d_x = GridQ4._num_elems1dh(shape)
        if e >= nelems1d_x:
            e1 = e - nelems1d_x
            x0 = e1 / (shape[1] - 1) + shape[0] * (e1 % (shape[1] - 1))
            return np.array([x0, x0, x0 + shape[0], x0 + shape[0]])
        else:
            x0 = e + e / nelems1d_x
            return np.array([x0, x0 + 1, x0 + 1, x0])

    @staticmethod
    def _num_elems1dh(shape):
        return (shape[0] - 1) * shape[1]

    @property
    def _elems1dh_nodes(self):
        return self.elems1d_nodes[:GridQ4._num_elems1dh(self.shape)]

    @property
    def _elems1dv_nodes(self):
        return self.elems1d_nodes[GridQ4._num_elems1dh(self.shape):]

    @staticmethod
    def _elem_nodes(e, num_elems_x):
        x0 = e + e / num_elems_x
        return np.array([x0, x0 + 1, x0 + num_elems_x + 2, x0 + num_elems_x + 1])

    def _inhline(self, n1, n2):
        return n1 / self.shape[0] == n2 / self.shape[0]

    def _invline(self, n1, n2):
        return n1 % self.shape[0] == n2 % self.shape[0]

    def find_elems_between(self, coords1, coords2):
        n1, n2 = [find_node(node, coords) for coords in [coords1, coords2]]
        if n1 == n2:
            return 0, [(n1, self.elem_coords(n1, 0))]
        elif self._inhline(n1, n2):
            e1, e2 = [find_elem_with_vertex(n, 0, self._elems1dh_nodes)
                      for n, pos in zip([n1, n2], [0, 1])]
            return 1, [(e, self.elem_coords(e, 1)) for e in range(e1, e2 + 1)]
        elif self._invline(n1, n2):
            e1, e2 = [GridQ4._num_elems1dh +
                      find_elem_with_vertex(n, 0, self._elems1dv_nodes)
                      for n, pos in zip([n1, n2], [0, 1])]
            return 1, [(e, self.elem_coords(e, 1)) for e in range(e1, e2 + 1)]
        else:
            e1, e2 = [find_elem_with_vertex(n, pos, self.elems_nodes)
                      for n, pos in zip([n1, n2], [0, 3])]
            nelemsx = self.shape[0] - 1
            return 2, [(i + j, self.elem_coords(i + j, 2))
                       for i in range(e1 / nelemsx, e2 / nelemsx + 1)
                       for j in range(e1 % nelemsx, e2 % nelemsx + 1)]

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
