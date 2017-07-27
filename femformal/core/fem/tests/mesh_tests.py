import logging
import unittest
from itertools import product as cartesian_product

import numpy as np

from .. import mesh as mesh


logger = logging.getLogger(__name__)

class TestGridQ4(unittest.TestCase):
    def setUp(self):
        self.L = 16
        self.c = 2
        self.xs = np.linspace(0, self.L, 5)
        self.ys = np.linspace(0, self.c, 3)
        # nodes_coords = np.array(sorted(cartesian_product(self.xs, self.ys),
        #                                key=lambda x: x[1]))
        nodes_coords = np.array(list(cartesian_product(self.xs, self.ys)))
        self.mesh = mesh.GridQ4(nodes_coords, (len(self.xs), len(self.ys)), None)


    def test_elem_nodes(self):
        e = 0
        num_elems_x = 4
        np.testing.assert_array_equal(mesh.GridQ4._elem_nodes(e, num_elems_x),
                                      [0, 1, 6, 5])
        e = 9
        np.testing.assert_array_equal(mesh.GridQ4._elem_nodes(e, num_elems_x),
                                      [11, 12, 17, 16])

    def test_num_elems1d_x(self):
        shape = (5, 3)
        np.testing.assert_equal(mesh.GridQ4._num_elems1dh(shape), 12)

    def test_elem1d_nodes(self):
        shape = (5, 3)
        e = 5
        np.testing.assert_array_equal(mesh.GridQ4._elem1d_nodes(e, shape),
                                      [6, 7, 7, 6])
        e = 19
        np.testing.assert_array_equal(mesh.GridQ4._elem1d_nodes(e, shape),
                                      [8, 8, 13, 13])

    def test_inlines(self):
        self.assertTrue(self.mesh._inhline(7, 8))
        self.assertFalse(self.mesh._inhline(7, 13))
        self.assertTrue(self.mesh._invline(1, 6))
        self.assertFalse(self.mesh._invline(7, 8))

    def test_find_elems_between_0d(self):
        c1 = np.array([8, 0])
        c2 = np.array([8, 0])
        expected = mesh.ElementSet(0, {2: np.array([c1, c1, c1, c1])})
        result = self.mesh.find_elems_between(c1, c2)
        np.testing.assert_equal(result, expected)

    def test_find_elems_between_1d_hor(self):
        c1 = np.array([8, 0])
        c2 = np.array([16, 0])
        c3 = np.array([12, 0])
        expected = mesh.ElementSet(1, {2: np.array([c1, c3, c3, c1]),
                                       3: np.array([c3, c2, c2, c3])})
        result = self.mesh.find_elems_between(c1, c2)
        np.testing.assert_equal(result, expected)

    def test_find_elems_between_1d_ver(self):
        c1 = np.array([8, 0])
        c2 = np.array([8, 2])
        c3 = np.array([8, 1])
        expected = mesh.ElementSet(1, {16: np.array([c1, c1, c3, c3]),
                                       17: np.array([c3, c3, c2, c2])})
        result = self.mesh.find_elems_between(c1, c2)
        np.testing.assert_equal(result, expected)

    def test_find_elems_between_2d(self):
        c1 = np.array([8, 0])
        c2 = np.array([16, 2])
        expected = set([2, 3, 6, 7])
        result = self.mesh.find_elems_between(c1, c2)
        np.testing.assert_equal(result.dimension, 2)
        np.testing.assert_equal(set(result.elems), expected)


class TestMeshGlobals(unittest.TestCase):
    def setUp(self):
        self.L = 16
        self.c = 2
        self.xs = np.linspace(0, self.L, 5)
        self.ys = np.linspace(0, self.c, 3)
        self.nodes_coords = np.array([[ 0,  0 ],
                                      [ 4,  0 ],
                                      [ 8,  0 ],
                                      [12,  0],
                                      [16,  0],
                                      [ 0,  1 ],
                                      [ 4,  1 ],
                                      [ 8,  1 ],
                                      [12,  1],
                                      [16,  1],
                                      [ 0,  2 ],
                                      [ 4,  2 ],
                                      [ 8,  2 ],
                                      [12,  2],
                                      [16,  2]])
        self.elems_nodes = np.array([[ 0,  1,  6,  5 ],
                                     [ 1,  2,  7,  6 ],
                                     [ 2,  3,  8,  7 ],
                                     [ 3,  4,  9,  8 ],
                                     [ 5,  6, 11, 10 ],
                                     [ 6,  7, 12, 11 ],
                                     [ 7,  8, 13, 12 ],
                                     [ 8,  9, 14, 13 ]])


    def test_find_elem_with_vertex(self):
        for elem, nodes in enumerate(self.elems_nodes):
            for i in range(len(nodes)):
                self.assertEqual(
                    mesh.find_elem_with_vertex(nodes[i], i, self.elems_nodes),
                    elem)

        with self.assertRaises(ValueError):
            mesh.find_elem_with_vertex(-1, 0, self.elems_nodes)

    def test_find_node(self):
        for n, coords in enumerate(self.nodes_coords):
            self.assertEqual(mesh.find_node(coords, self.nodes_coords), n)
        with self.assertRaises(ValueError):
            mesh.find_node(np.array([-50, -50]), self.nodes_coords)

    def test_flatten_coord(self):
        l = 0
        a, b, c = 5, 3, 4
        shape = (a,b,c)
        for k in range(c):
            for j in range(b):
                for i in range(a):
                    np.testing.assert_equal(
                        mesh._flatten_coord([i,j,k], shape), l)
                    np.testing.assert_array_equal(
                        mesh._unflatten_coord(l, shape), [i,j,k])
                    l += 1

class TestGridMesh(unittest.TestCase):
    def setUp(self):
        xs = np.linspace(0, 8, 5)
        ys = np.linspace(0, 6, 3)
        zs = np.linspace(0, 6, 4)
        # nodes_coords = np.array(sorted(cartesian_product(self.xs, self.ys),
        #                                key=lambda x: x[1]))
        nodes_coords = np.array(list(cartesian_product(xs, ys, zs)))
        self.mesh = mesh.GridMesh(nodes_coords, (len(xs), len(ys), len(zs)), None)


    def test_find_nodes_between(self):
        c1 = np.array([2, 3, 2])
        c2 = np.array([4, 6, 4])
        expected = [21, 22, 26, 27, 36, 37, 41, 42]
        np.testing.assert_array_equal(self.mesh.find_nodes_between(c1, c2), expected)

class TestMesh(unittest.TestCase):
    def setUp(self):
        self.xs = np.linspace(0, 8, 5)
        self.ys = np.linspace(0, 6, 3)
        self.zs = np.linspace(0, 6, 4)
        nodes_coords = np.array(list(cartesian_product(self.xs, self.ys, self.zs)))
        self.mesh = mesh.Mesh(nodes_coords, None)


    def test_sorted_nodes(self):
        nodes = self.mesh.nodes_coords
        l = 0
        for k in range(len(self.zs)):
            for j in range(len(self.ys)):
                for i in range(len(self.xs)):
                    np.testing.assert_array_equal(
                        nodes[l], [self.xs[i], self.ys[j], self.zs[k]])
                    l += 1
