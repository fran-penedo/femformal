import unittest
from itertools import product as cartesian_product

import numpy as np
from numpy import testing as npt
from stlmilp import stl as stl

from .. import logic
from ..fem import mesh, element


class test_logic(unittest.TestCase):
    def setUp(self):
        self.apc1 = logic.APCont(np.array([0, 2]), "<", lambda x: x, lambda x: 1, uderivs=0)
        self.apc2 = logic.APCont(np.array([1, 4]), ">", lambda x: x, lambda x: 1, uderivs=1)
        self.apd1 = logic.APDisc(1, {1: [5.0, 10.0], 2: [6.0, 10.0]}, False, uderivs=1)
        self.xpart = np.linspace(0, 5, 11)
        self.apd1_string = "((y 1 1 < 5.0 10.0) & (y 1 2 < 6.0 10.0))"
        self.form = "G_[0, 1] ({})".format(self.apd1_string)
        self.signal1 = logic.SysSignal(1, stl.GT, 5.0, 10.0, False, 1, xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000])
        self.signal2 = logic.SysSignal(1, stl.LE, 5.0, 10.0, False, 0, xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000])
        self.signal3 = logic.SysSignal(1, stl.LE, 5.0, 10.0, True, 0, xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000])

        self.L = 16
        self.c = 2
        self.xs = np.linspace(0, self.L, 5)
        self.ys = np.linspace(0, self.c, 3)
        nodes_coords = np.array(sorted(cartesian_product(self.xs, self.ys),
                                       key=lambda x: x[1]))
        self.mesh = mesh.GridQ4(nodes_coords, (len(self.xs), len(self.ys)))
        self.build_elem = element.BLQuadQ4
        self.signal4 = logic.SysSignal(1, stl.GT, 5.0, 10.0, False, 0, region_dim=2,
                             mesh_=self.mesh, build_elem=element.BLQuadQ4)


    def test_apc_to_apd(self):
        apd = logic.ap_cont_to_disc(self.apc1, self.xpart)
        els = [0,1,2,3]
        ps = np.array([[.25, .75, 1.25, 1.75], [1.0, 1.0, 1.0, 1.0]]).T
        self.assertEqual(apd.r, 1)
        self.assertEqual(set(apd.m.keys()), set(els))
        np.testing.assert_array_almost_equal([apd.m[e] for e in els], ps)

        apd = logic.ap_cont_to_disc(self.apc2, self.xpart)
        els = [2,3,4,5,6,7]
        ps = np.array([[1.25, 1.75, 2.25, 2.75, 3.25, 3.75], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).T
        self.assertEqual(apd.r, -1)
        self.assertEqual(set(apd.m.keys()), set(els))
        np.testing.assert_array_almost_equal([apd.m[e] for e in els], ps)

    def test_apd_string(self):
        self.assertEqual(str(self.apd1), self.apd1_string)

    def test_expr_parser(self):
        fdt_mult = 2
        bounds = [-1000, 1000]
        parser = logic.stl_parser(self.xpart, fdt_mult, bounds)
        form = parser.parseString(self.apd1_string)[0]
        s1, s2 = [f.args[0] for f in form.args]

        self.assertEqual(str(s1), "y 1 1 < 5.0 10.0")
        self.assertEqual(str(s2), "y 1 2 < 6.0 10.0")
        self.assertEqual(s1.fdt_mult, fdt_mult)
        self.assertEqual(s1.bounds, bounds)
        self.assertEqual(s2.fdt_mult, fdt_mult)
        self.assertEqual(s2.bounds, bounds)

    def test_expr_parser_2d(self):
        fdt_mult = 2
        bounds = [-1000, 1000]
        parser = logic.stl_parser(None, fdt_mult, bounds, self.mesh, self.build_elem)
        pred = "2 0 0 1 > 5.0 10.0"
        form = parser.parseString("(" + pred + ")")[0]
        s4 = form.args[0]

        self.assertEqual(str(s4), pred)
        self.assertEqual(s4.fdt_mult, fdt_mult)
        self.assertEqual(s4.bounds, bounds)

    def test_syssignal(self):
        self.assertEqual(self.signal1.labels[0](5), "d_1_5")
        self.assertEqual(self.signal1.labels[1](5), "d_2_5")
        self.assertEqual(self.signal1.f([2.0, 4.0]), -1.0)
        self.assertEqual(self.signal2.f([2.0, 4.0]), 2.0)
        self.assertEqual(self.signal3.f([2.0]), 3.0)

    def test_signal_perturb(self):
        self.signal1.perturb(lambda a, b, c, d: 1.0)
        self.assertEqual(self.signal1.f([2.0, 4.0]), -2.0)
        self.signal2.perturb(lambda a, b, c, d: 1.0)
        self.assertEqual(self.signal2.f([2.0, 4.0]), 1.0)
        self.signal4.perturb(lambda a, b, c, d: 1.0)
        self.assertEqual(self.signal4.f([1,2,3,4]), -3.5)

    def test_scale_time(self):
        fdt_mult = 2
        bounds = [-1000, 1000]
        parser = logic.stl_parser(self.xpart, fdt_mult, bounds)
        form = parser.parseString(self.form)[0]
        dt = .5
        logic.scale_time(form, dt)
        self.assertEqual(form.bounds, [0, 2])

    def test_ap_cont2d_to_disc(self):
        pred = lambda x, y: x + y
        dpred = lambda x, y: np.array([1, 1])
        region = np.array([[8, 0], [16, 1]])
        apc = logic.APCont2D(0, region, '>', pred, dpred)
        expected_map = {2: (10.5, np.array([1,1])),
                        3: (14.5, np.array([1,1]))}
        apd = logic.ap_cont2d_to_disc(apc, self.mesh, element.BLQuadQ4)
        npt.assert_equal(apd.r, apc.r)
        npt.assert_equal(apd.region_dim, 2)
        self.assertFalse(apd.isnode)
        self.assertSetEqual(set(apd.m.keys()), set(expected_map.keys()))
        for k in apd.m.keys():
            npt.assert_almost_equal(apd.m[k][0], expected_map[k][0])
            npt.assert_array_almost_equal(apd.m[k][1], expected_map[k][1])

    def test_syssignal_2d_full(self):
        s1 = logic.SysSignal(1, stl.GT, 5.0, 10.0, False, 0, region_dim=2,
                             mesh_=self.mesh, build_elem=element.BLQuadQ4)
        expected_labels = ["d_2_5", "d_4_5", "d_14_5", "d_12_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [1,2,3,4]
        expected_f = np.mean(vs) - 5.0
        npt.assert_equal(s1.f(vs), expected_f)

    def test_syssignal_2d_hor(self):
        s1 = logic.SysSignal(1, stl.LE, 5.0, 10.0, False, 0, region_dim=1,
                             mesh_=self.mesh, build_elem=element.BLQuadQ4)
        expected_labels = ["d_2_5", "d_4_5", "d_4_5", "d_2_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [1,2,2,1]
        expected_f = 5.0 - np.mean(vs)
        npt.assert_equal(s1.f(vs), expected_f)

    def test_syssignal_2d_ver(self):
        s1 = logic.SysSignal(14, stl.LE, 5.0, 10.0, False, 0, region_dim=1,
                             u_comp=1, mesh_=self.mesh, build_elem=element.BLQuadQ4)
        expected_labels = ["d_3_5", "d_3_5", "d_13_5", "d_13_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [1,1,2,2]
        expected_f = 5.0 - np.mean(vs)
        npt.assert_equal(s1.f(vs), expected_f)

    def test_syssignal_2d_point(self):
        s1 = logic.SysSignal(5, stl.LE, 5.0, 10.0, False, 0, region_dim=0,
                             u_comp=1, mesh_=self.mesh, build_elem=element.BLQuadQ4)
        expected_labels = ["d_11_5", "d_11_5", "d_11_5", "d_11_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [2,2,2,2]
        expected_f = 5.0 - np.mean(vs)
        npt.assert_equal(s1.f(vs), expected_f)
