import unittest

import numpy as np

from .. import fem_util as fem


class test_fem_util(unittest.TestCase):
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
                    fem.find_elem_with_vertex(nodes[i], i, self.elems_nodes),
                    elem)

        with self.assertRaises(ValueError):
            fem.find_elem_with_vertex(-1, 0, self.elems_nodes)

    def test_find_node(self):
        for n, coords in enumerate(self.nodes_coords):
            self.assertEqual(fem.find_node(coords, self.nodes_coords), n)
        with self.assertRaises(ValueError):
            fem.find_node(np.array([-50, -50]), self.nodes_coords)

    def test_elem_nodes(self):
        e = 0
        num_elems_x = 4
        np.testing.assert_array_equal(fem.GridQ4._elem_nodes(e, num_elems_x),
                                      [0, 1, 6, 5])
        e = 9
        np.testing.assert_array_equal(fem.GridQ4._elem_nodes(e, num_elems_x),
                                      [11, 12, 17, 16])

