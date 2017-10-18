"""
FEM mesh data structures and definitions
"""
from __future__ import division, absolute_import, print_function

import logging
import itertools

import numpy as np


logger = logging.getLogger(__name__)

class Mesh(object):
    def __init__(self, nodes_coords, elem_shape):
        nodes_coords = nodes_coords[np.lexsort(nodes_coords.T)]
        self.nodes_coords = nodes_coords
        self.nelems = np.prod(elem_shape)
        self.elem_shape = elem_shape

    @property
    def nnodes(self):
        return self.nodes_coords.shape[0]

    def _interpolate(self, d, fn):
        d = self._reshape_int_values(d)

        def _interp(*args):
            x = np.array(args)
            e = self.find_containing_elem(x)
            d_elem = d.take(self.elem_nodes(e), axis=-2)
            return getattr(self.elements[e], fn)(d_elem, x)

        return _interp

    def _reshape_int_values(self, d):
        if d.shape[-1] % self.nnodes != 0:
            raise ValueError("Interpolating values do not agree with number of "
                            "nodes: nvalues = {}, nnnodes = {}".format(
                                d.shape[-1], self.nnodes))
        dofs = d.shape[-1] // self.nnodes
        if len(d.shape) == 1:
            ret = d.reshape(self.nnodes, dofs)
        else:
            if dofs > 1:
                ret = d.reshape(d.shape[0], self.nnodes, dofs).T
            else:
                ret = d.T

        return ret

    def interpolate(self, d):
        return self._interpolate(d, 'interpolate_phys')

    def interpolate_derivatives(self, d):
        if len(d.shape) > 1:
            raise ValueError("Calling non vectorized function with vector")
        return self._interpolate(d, 'interpolate_derivatives_phys')

    def interpolate_strain(self, d):
        dofs = d.shape[-1] // self.nnodes
        logger.debug(d.shape)
        if dofs != 2:
            raise ValueError("Need dofs = 2 for strain interpolation. Given {}".format(dofs))
        return self._interpolate(d, 'interpolate_strain')

    def max_diffs(self, d):
        d = _reshape_int_values(d)

    def elem_nodes(self, elem, dim):
        raise NotImplementedError()

    @property
    def build_elem(self):
        return self._build_elem

    @build_elem.setter
    def build_elem(self, value):
        self._build_elem = value
        if self._build_elem is not None:
            self.elements = [self.build_elem(self.elem_coords(e))
                            for e in range(self.nelems)]
        else:
            self.elements = None

    def elem_coords(self, elem, dim=2):
        return np.array([self.nodes_coords[n] for n in self.elem_nodes(elem, dim)])

    def get_elem(self, elem, dim=2):
        return self.build_elem(self.elem_coords(elem, dim))

    def find_containing_elem(self, coords):
        raise NotImplementedError()

    def find_near_node(self, coords, position):
        try:
            n = find_node(coords, self.nodes_coords)
        except:
            e = self.find_containing_elem(coords)
            n = self.elem_nodes(e)[position]
        return n


class GridMesh(Mesh):
    def __init__(self, partitions, elem_shape):
        nodes_coords = np.array(list(itertools.product(*partitions)))
        Mesh.__init__(self, nodes_coords, elem_shape)
        self.partitions = partitions
        shape = np.array([len(part) for part in partitions])
        if np.prod(shape) != self.nnodes:
            raise ValueError(
                "Shape {} not compatible with number of nodes {}".format(
                    shape, self.nnodes))
        self.shape = shape

    def find_nodes_between(self, coords1, coords2):
        n1, n2 = [find_node(coords, self.nodes_coords)
                  for coords in [coords1, coords2]]
        n1s, n2s = [np.array(_unflatten_coord(n, self.shape)) for n in [n1, n2]]
        nodes = [_flatten_coord(n1s + off, self.shape) for off in
                itertools.product(
                    *[range(n2s[i] - n1s[i] + 1) for i in range(len(n1s))])]
        return ElementSet(0, {n: self.nodes_coords[n] for n in nodes})

    def connected_fwd(self, node):
        conn = []
        mesh_coord = np.array(_unflatten_coord(node, self.shape))
        for i in range(len(self.shape)):
            if mesh_coord[i] != self.shape[i] - 1:
                conn.append(_flatten_coord(
                    [1 if x == i else 0 for x in range(mesh_coord.shape[0])] +
                    mesh_coord, self.shape))
        return conn

    def find_elems_covering(self, coords1, coords2):
        n1 = self.find_near_node(coords1, 0)
        n2 = self.find_near_node(coords2, 2)
        return self.find_elems_between(self.nodes_coords[n1], self.nodes_coords[n2])

    def find_containing_elem(self, coords):
        for e in range(self.nelems):
            left, right = [self.nodes_coords[self.elems_nodes[e][p]]
                           for p in [0, 2]]
            if np.all(coords >= left) and np.all(coords <= right):
                return e
        raise ValueError(
            "Coordinates outside the domain. Given {}".format(coords))


    def find_border_full_elems(self):
        elem_shape = self.elem_shape
        border_coords = [list(range(sh)) for sh in elem_shape]
        elems = []
        for axis in range(len(border_coords)):
            for i in [0, -1]:
                coords_list = ([border_coords[j][1:-1] for j in range(axis)] +
                               [[border_coords[axis][i]]] +
                               [border_coords[j] for j in
                                range(axis + 1, len(border_coords))])
                for elem in itertools.product(*coords_list):
                    elems.append(_flatten_coord(elem, elem_shape))
        return elems


class GridMesh2D(GridMesh):
    def __init__(self, partitions, elem_shape):
        if len(partitions) != 2 or len(elem_shape) != 2:
            raise ValueError("GridMesh2D must be constructed over a 2D domain")
        GridMesh.__init__(self, partitions, elem_shape)

    def _inhline(self, n1, n2):
        return n1 // self.shape[0] == n2 // self.shape[0]

    def _invline(self, n1, n2):
        return n1 % self.shape[0] == n2 % self.shape[0]

    def find_elems_between(self, coords1, coords2):
        n1, n2 = [find_node(coords, self.nodes_coords)
                  for coords in [coords1, coords2]]
        if n1 == n2:
            return ElementSet(0, {n1: self.elem_coords(n1, 0)})
        elif self._inhline(n1, n2):
            try:
                e1, e2 = [find_elem_with_vertex(n, pos, self._elems1dh_nodes)
                        for n, pos in zip([n1, n2], [0, 2])]
            except ValueError as e:
                logger.error("Given coordinates not aligned with element "
                             "partition. Given {}".format([coords1, coords2]))
                raise e
            return ElementSet(1, {e: self.elem_coords(e, 1)
                                for e in range(e1, e2 + 1)})
        elif self._invline(n1, n2):
            try:
                e1, e2 = [self._num_elems1dh(self.shape) +
                        find_elem_with_vertex(n, pos, self._elems1dv_nodes)
                        for n, pos in zip([n1, n2], [0, 2])]
            except ValueError as e:
                logger.error("Given coordinates not aligned with element "
                             "partition. Given {}".format([coords1, coords2]))
                raise e
            return ElementSet(1, {e: self.elem_coords(e, 1)
                                for e in range(e1, e2 + 1)})
        else:
            try:
                e1, e2 = [find_elem_with_vertex(n, pos, self.elems_nodes)
                        for n, pos in zip([n1, n2], [0, 2])]
            except ValueError as e:
                logger.error("Given coordinates not aligned with element "
                             "partition. Given {}".format([coords1, coords2]))
                raise e
            nelemsx = self.elem_shape[0]
            return ElementSet(
                2, {i * nelemsx + j: self.elem_coords(i * nelemsx + j, 2)
                    for i in range(e1 // nelemsx, e2 // nelemsx + 1)
                    for j in range(e1 % nelemsx, e2 % nelemsx + 1)})

    def find_2d_containing_elems(self, elem, dim=2):
        """Returns all full dimension elements containing the element"""

        if dim == 2:
            return ElementSet(2, {elem: self.elem_coords(elem)})
        elif dim == 1:
            ns = self.elem_nodes(elem, dim)
            return self.find_2d_containing_elems(ns[0], dim=0).intersection(
                self.find_2d_containing_elems(ns[2], dim=0))
        elif dim == 0:
            elem_coords_map = {}
            for i in range(4):
                try:
                    cont_elem = find_elem_with_vertex(elem, i, self.elems_nodes)
                    elem_coords_map[cont_elem] = self.elem_coords(cont_elem)
                except ValueError:
                    pass
            return ElementSet(2, elem_coords_map)


def _unflatten_coord(x, shape):
    i = x % shape[0]
    if len(shape) == 1:
        return (i, )
    else:
        return (i, ) + _unflatten_coord((x - i) // shape[0], shape[1:])

def _flatten_coord(xs, shape):
    if len(shape) == 1:
        return xs[0]
    else:
        return xs[0] + shape[0] * _flatten_coord(xs[1:], shape[1:])


class GridQ4(GridMesh2D):
    def __init__(self, partitions, build_elem):
        elem_shape = [len(partitions[0]) - 1, len(partitions[1]) - 1]
        GridMesh.__init__(self, partitions, elem_shape)
        self.elems_nodes = np.array([GridQ4._elem_nodes(e, self.shape[0] - 1)
                                for e in range(self.nelems)])
        self.elems1d_nodes = np.array(
            [GridQ4._elem1d_nodes(e, self.shape)
             for e in range(2 * self.nnodes - np.sum(self.shape))])
        self.build_elem = build_elem

    def elem_nodes(self, elem, dim=2):
        if dim == 0:
            return [elem for i in range(4)]
        elif dim == 1:
            return self.elems1d_nodes[elem]
        elif dim == 2:
            return self.elems_nodes[elem]
        else:
            raise NotImplementedError(
                "Dimension ({}) must be between 0 and 2".format(dim))

    @staticmethod
    def _elem1d_nodes(e, shape):
        nelems1dh = GridQ4._num_elems1dh(shape)
        nelems1d_x = shape[0] - 1
        if e >= nelems1dh:
            e1 = e - nelems1dh
            nelems1d_y = shape[1] - 1
            x0 = e1 // nelems1d_y + shape[0] * (e1 % nelems1d_y)
            return np.array([x0, x0, x0 + shape[0], x0 + shape[0]])
        else:
            x0 = e + e // nelems1d_x
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
        x0 = e + e // num_elems_x
        return np.array([x0, x0 + 1, x0 + num_elems_x + 2, x0 + num_elems_x + 1])

    def _inhline(self, n1, n2):
        return n1 // self.shape[0] == n2 // self.shape[0]

    def _invline(self, n1, n2):
        return n1 % self.shape[0] == n2 % self.shape[0]

    def find_near_node(self, coords, position):
        try:
            n = find_node(coords, self.nodes_coords)
        except:
            e = self.find_containing_elem(coords)
            n = self.elem_nodes(e)[position]
        return n

    # def find_containing_elem(self, coords):
    #     """Returns an arbitrary element containing the point"""
    #
    #     node_mesh_coords = [bisect_right(self.partitions[i], coords[i]) - 1
    #                         for i in range(len(self.partitions))]
    #
    #     position = 0
    #     if node_mesh_coords[0] == len(self.partitions[0]) - 1:
    #         position += 1
    #     if node_mesh_coords[1] == len(self.partitions[1]) - 1:
    #         position +=3
    #     if position > 3:
    #         position = 2
    #
    #     vnode = _flatten_coord(node_mesh_coords, self.shape)
    #     return find_elem_with_vertex(vnode, position, self.elems_nodes)

    def find_2d_containing_elems(self, elem, dim=2):
        """Returns all full dimension elements containing the element"""

        if dim == 2:
            return ElementSet(2, {elem: self.elem_coords(elem)})
        elif dim == 1:
            ns = self.elem_nodes(elem, dim)
            return self.find_2d_containing_elems(ns[0], dim=0).intersection(
                self.find_2d_containing_elems(ns[2], dim=0))
        elif dim == 0:
            elem_coords_map = {}
            for i in range(4):
                try:
                    cont_elem = find_elem_with_vertex(elem, i, self.elems_nodes)
                    elem_coords_map[cont_elem] = self.elem_coords(cont_elem)
                except ValueError:
                    pass
            return ElementSet(2, elem_coords_map)

    def find_border_elems(self):
        bottom = list(range(self.shape[0] - 1))
        top = [(self.shape[0] - 1) * (self.shape[1] - 1) + e for e in bottom]
        left = [self._num_elems1dh(self.shape) + i for i in range(self.shape[1] - 1)]
        right = [len(self.elems1d_nodes) - i for i in range(1, self.shape[1])]
        return bottom + top + left + right


class GridQ9(GridMesh2D):
    def __init__(self, partitions, build_elem):
        elem_shape = [(len(partitions[0]) - 1) // 2, (len(partitions[1]) - 1) // 2]
        GridMesh.__init__(self, partitions, elem_shape)
        self.elems_nodes = np.array([
            GridQ9._elem_nodes(e, (self.shape[0] - 1) // 2)
            for e in range(self.nelems)])
        self.elems1d_nodes = np.array(
            [GridQ9._elem1d_nodes(e, self.shape)
             for e in range(self.nnodes - np.sum(self.shape) // 2)])
        self.build_elem = build_elem

    def elem_nodes(self, elem, dim=2):
        if dim == 0:
            return [elem for i in range(9)]
        elif dim == 1:
            return self.elems1d_nodes[elem]
        elif dim == 2:
            return self.elems_nodes[elem]
        else:
            raise NotImplementedError(
                "Dimension ({}) must be between 0 and 2".format(dim))

    @staticmethod
    def _elem1d_nodes(e, shape):
        nelems1dh = GridQ9._num_elems1dh(shape)
        nelems1d_x = (shape[0] - 1) // 2
        if e >= nelems1dh:
            e1 = e - nelems1dh
            nelems1d_y = (shape[1] - 1) // 2
            x0 = e1 // nelems1d_y + 2 * shape[0] * (e1 % nelems1d_y)
            return np.array([x0, x0, x0 + 2 * shape[0], x0 + 2 * shape[0],
                             x0, x0 + shape[0], x0 + 2 * shape[0],
                             x0 + shape[0], x0 + shape[0]])
        else:
            x0 = 2 * e + e // nelems1d_x
            return np.array([x0, x0 + 2, x0 + 2, x0,
                             x0 + 1, x0 + 2, x0 + 1, x0, x0 + 1])

    @staticmethod
    def _num_elems1dh(shape):
        return (shape[0] - 1) * shape[1] // 2

    @property
    def _elems1dh_nodes(self):
        return self.elems1d_nodes[:GridQ9._num_elems1dh(self.shape)]

    @property
    def _elems1dv_nodes(self):
        return self.elems1d_nodes[GridQ9._num_elems1dh(self.shape):]

    @staticmethod
    def _elem_nodes(e, num_elems_x):
        nnodes_x = num_elems_x * 2 + 1
        x0 = 2 * (e % num_elems_x) + (e // num_elems_x) * nnodes_x * 2
        return np.array([x0, x0 + 2, x0 + nnodes_x * 2 + 2, x0 + nnodes_x * 2,
                         x0 + 1, x0 + nnodes_x + 2, x0 + nnodes_x * 2 + 1,
                         x0 + nnodes_x, x0 + nnodes_x + 1])

    def find_border_elems(self):
        bottom = list(range(self.elem_shape[0]))
        top = [self.elem_shape[0] * (self.shape[1] - 1) + e for e in bottom]
        left = [self._num_elems1dh(self.shape) + i for i in range(self.elem_shape[1])]
        right = [len(self.elems1d_nodes) - i for i in range(1, self.elem_shape[1] + 1)]
        return bottom + top + left + right


class ElementSet(object):
    def __init__(self, dimension, elem_coords_map):
        self.dimension = dimension
        self.elem_coords_map = elem_coords_map

    @property
    def elems(self):
        return sorted(self.elem_coords_map.keys())

    def intersection(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Intersection undefined for sets of elements of different dimension")
        else:
            elem_coords_map = {
                e: self[e] for e in self.elem_coords_map.viewkeys() &
                other.elem_coords_map.viewkeys()}
            return ElementSet(self.dimension, elem_coords_map)

    def __getitem__(self, el):
        return self.elem_coords_map[el]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.dimension != other.dimension or \
                    len(self.elems) != len(other.elems):
                return False
            for el in self.elems:
                if el not in other.elems or \
                        not np.all(np.isclose(self[el], other[el])):
                    return False
            return True
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __iter__(self):
        return iter(self.elem_coords_map.items())


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
