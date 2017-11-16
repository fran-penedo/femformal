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
        uderivs = 1
        region_dim = 1
        isnode = False
        r = 1
        self.apd1 = logic._APDisc([
            logic.STLPred(1, r, 5.0, 10.0, isnode, uderivs=uderivs, region_dim=region_dim),
            logic.STLPred(2, r, 6.0, 10.0, isnode, uderivs=uderivs, region_dim=region_dim)
        ])
        self.xpart = np.linspace(0, 5, 11)
        self.apd1_string = "((1 0 1 1 0 < 5.0 10.0 -1) & (1 0 1 2 0 < 6.0 10.0 -1))"
        self.form = "G_[0, 1] ({})".format(self.apd1_string)
        self.signal1 = logic.SysSignal(
            logic.STLPred(1, -1, 5.0, 10.0, isnode=False, uderivs=1, region_dim=1),
            xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000])
        self.signal2 = logic.SysSignal(
            logic.STLPred(1, 1, 5.0, 10.0, isnode=False, uderivs=0, region_dim=1),
            xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000])
        self.signal3 = logic.SysSignal(
            logic.STLPred(1, 1, 5.0, 10.0, isnode=True, uderivs=0, region_dim=0),
            xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000])
        # self.signal2 = logic.SysSignal(1, stl.LE, 5.0, 10.0, False, 0, xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000], region_dim=1)
        # self.signal3 = logic.SysSignal(1, stl.LE, 5.0, 10.0, True, 0, xpart=self.xpart, fdt_mult=2, bounds=[-1000, 1000], region_dim=0)

        self.L = 16
        self.c = 2
        self.xs = np.linspace(0, self.L, 5)
        self.ys = np.linspace(0, self.c, 3)
        # nodes_coords = np.array(list(cartesian_product(self.xs, self.ys)))
        self.mesh = mesh.GridQ4([self.xs, self.ys], element.BLQuadQ4)
        # self.build_elem = element.BLQuadQ4
        # self.signal4 = logic.SysSignal(1, stl.GT, 5.0, 10.0, False, 0, region_dim=2,
        #                      mesh_=self.mesh, build_elem=element.BLQuadQ4)
        self.signal4 = logic.SysSignal(
            logic.STLPred(1, -1, 5.0, 10.0, False, uderivs=0, region_dim=2),
            mesh_=self.mesh)


    def test_apc_to_apd(self):
        apd = logic._ap_cont_to_disc(self.apc1, self.xpart)
        els = [0,1,2,3]
        ps = np.array([[.25, .75, 1.25, 1.75], [1.0, 1.0, 1.0, 1.0]]).T
        self.assertEqual(len(apd.stlpred_list), len(els))
        for stlpred in apd.stlpred_list:
            self.assertEqual(stlpred.r, 1)
        self.assertEqual(set(stlpred.index for stlpred in apd.stlpred_list),
                         set(els))
        np.testing.assert_array_almost_equal(
            [[stlpred.p, stlpred.dp]
             for stlpred in sorted(apd.stlpred_list, key=lambda x: x.index)], ps)

        apd = logic._ap_cont_to_disc(self.apc2, self.xpart)
        els = [2,3,4,5,6,7]
        ps = np.array([[1.25, 1.75, 2.25, 2.75, 3.25, 3.75], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).T
        for stlpred in apd.stlpred_list:
            self.assertEqual(stlpred.r, -1)
        self.assertEqual(len(apd.stlpred_list), len(els))
        np.testing.assert_array_almost_equal(
            [[stlpred.p, stlpred.dp]
             for stlpred in sorted(apd.stlpred_list, key=lambda x: x.index)], ps)

    def test_apd_string(self):
        self.assertEqual(str(self.apd1), self.apd1_string)

    def test_expr_parser(self):
        fdt_mult = 2
        bounds = [-1000, 1000]
        parser = logic.stl_parser(self.xpart, fdt_mult, bounds)
        form = parser.parseString(self.apd1_string)[0]
        s1, s2 = [f.args[0] for f in form.args]

        self.assertEqual(str(s1), "(1 0 1 1 0 < 5.0 10.0 -1)")
        self.assertEqual(str(s2), "(1 0 1 2 0 < 6.0 10.0 -1)")
        self.assertEqual(s1.fdt_mult, fdt_mult)
        self.assertEqual(s1.bounds, bounds)
        self.assertEqual(s2.fdt_mult, fdt_mult)
        self.assertEqual(s2.bounds, bounds)

    def test_expr_parser_2d(self):
        fdt_mult = 2
        bounds = [-1000, 1000]
        parser = logic.stl_parser(None, fdt_mult, bounds, self.mesh)
        pred = "(2 0 0 1 0 > 5.0 10.0 -1)"
        form = parser.parseString(pred)[0]
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
        self.signal1.perturb(lambda x: 1.0)
        self.assertEqual(self.signal1.f([2.0, 4.0]), -2.0)
        self.signal2.perturb(lambda x: 1.0)
        self.assertEqual(self.signal2.f([2.0, 4.0]), 1.0)
        self.signal4.perturb(lambda x: 1.0)
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
        apd = logic._ap_cont2d_to_disc(apc, self.mesh)
        for pred in apd.stlpred_list:
            npt.assert_equal(pred.r, apc.r)
            npt.assert_equal(pred.region_dim, 2)
            self.assertFalse(pred.isnode)
            npt.assert_almost_equal(pred.p, expected_map[pred.index][0])
            npt.assert_array_almost_equal(pred.dp, expected_map[pred.index][1])

    def test_syssignal_2d_full(self):
        s1 = logic.SysSignal(
            logic.STLPred(1, -1, 5.0, 10.0, False, uderivs=0, region_dim=2),
            mesh_=self.mesh)
        expected_labels = ["d_2_5", "d_4_5", "d_14_5", "d_12_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [1,2,3,4]
        expected_f = np.mean(vs) - 5.0
        npt.assert_equal(s1.f(vs), expected_f)

    def test_syssignal_2d_full_stress(self):
        s1 = logic.SysSignal(
            logic.STLPred(
                1, -1, 5.0, 10.0, False, uderivs=1, region_dim=2, u_comp=1,
                query_point=0, deriv=1),
            mesh_=self.mesh)
        expected_labels = ["d_2_5", "d_3_5", "d_4_5", "d_5_5", "d_14_5", "d_15_5", "d_12_5", "d_13_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [1,2,3,4,1,6,1,4]
        expected_f = 2 / (self.c / (len(self.ys) - 1)) - 5.0
        npt.assert_equal(s1.f(vs), expected_f)

    def test_syssignal_2d_hor(self):
        s1 = logic.SysSignal(
            logic.STLPred(1, 1, 5.0, 10.0, False, uderivs=0, region_dim=1),
            mesh_=self.mesh)
        expected_labels = ["d_2_5", "d_4_5", "d_4_5", "d_2_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [1,2,2,1]
        expected_f = 5.0 - np.mean(vs)
        npt.assert_equal(s1.f(vs), expected_f)

    def test_syssignal_2d_ver(self):
        s1 = logic.SysSignal(
            logic.STLPred(14, 1, 5.0, 10.0, False, uderivs=0, region_dim=1, u_comp=1),
            mesh_=self.mesh)
        expected_labels = ["d_3_5", "d_3_5", "d_13_5", "d_13_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [1,1,2,2]
        expected_f = 5.0 - np.mean(vs)
        npt.assert_equal(s1.f(vs), expected_f)

    def test_syssignal_2d_point(self):
        s1 = logic.SysSignal(
            logic.STLPred(5, 1, 5.0, 10.0, False, uderivs=0, region_dim=0, u_comp=1),
            mesh_=self.mesh)
        expected_labels = ["d_11_5", "d_11_5", "d_11_5", "d_11_5"]
        self.assertListEqual([l(5) for l in s1.labels], expected_labels)
        vs = [2,2,2,2]
        expected_f = 5.0 - np.mean(vs)
        npt.assert_equal(s1.f(vs), expected_f)
