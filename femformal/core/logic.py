import itertools as it
from bisect import bisect_left, bisect_right

import numpy as np
import stlmilp.stl as stl
from pyparsing import Word, Literal, MatchFirst, nums, alphas

from . import system as sys
from .util import state_label, label, unlabel
from .fem import foobar as fem


class APCont(object):
    def __init__(self, A, r, p, dp = None, uderivs = 0):
        # A : [x_min, x_max] (np.array)
        self.A = A
        # r == 1: f < p, r == -1: f > p
        if r == "<":
            self.r = 1
        elif r == ">":
            self.r = -1
        else:
            self.r = r
        self.p = p
        self.uderivs = uderivs
        if dp:
            self.dp = dp
        else:
            self.dp = lambda x: 0


class APCont2D(APCont):
    def __init__(self, u_comp, A, r, p, dp):
        APCont.__init__(self, A, r, p, dp)
        self.v_comp = u_comp


class APDisc(object):
    def __init__(self, r, m, isnode = True, uderivs = 0, region_dim = 0):
        # r == 1: f < p, r == -1: f > p
        self.r = r
        if not isnode or region_dim == 0:
            self.isnode = isnode
            self.region_dim = region_dim
        else:
            self.region_dim = region_dim
            self.isnode = bool(region_dim)

        # m : i -> (p((x_i + x_{i+1})/2) if not isnode else i -> p(x_i),
        # dp(.....))
        self.m = m
        self.uderivs = uderivs

    def __str__(self):
        return "({})".format(" & ".join(
            ["({isnode} {uderivs} {index} {op} {p} {dp})".format(
                isnode="d" if self.isnode else "y", uderivs=self.uderivs,
                index=i, op="<" if self.r == 1 else ">",
                p=p, dp=dp) for (i, (p, dp)) in self.m.items()]))


def subst_spec_labels_disc(spec, regions):
    res = spec
    for k, v in regions.items():
        replaced = str(v)
        res = res.replace(k, replaced)
    return res

def ap_cont_to_disc(apcont, xpart):
    # xpart : [x_i] (list)
    # FIXME TS based approach probably has wrong index assumption
    r = apcont.r
    N1 = len(xpart)
    if apcont.A[0] == apcont.A[1]:
        if apcont.uderivs > 0:
            raise Exception("Derivatives at nodes are not well defined")
        i = min(max(bisect_left(xpart, apcont.A[0]), 0), N1 - 1)
        m = {i - 1: (apcont.p(xpart[i]), apcont.dp(xpart[i]))}
        isnode = True
    else:
        if apcont.uderivs > 1:
            raise Exception(
                ("Second and higher order derivatives are 0 for linear "
                "interpolation: uderivs = {}").format(apcont.uderivs))

        i_min = max(bisect_left(xpart, apcont.A[0]), 0)
        i_max = min(bisect_left(xpart, apcont.A[1]), N1 - 1)
        m = {i : (apcont.p((xpart[i] + xpart[i+1]) / 2.0),
                  apcont.dp((xpart[i] + xpart[i+1]) / 2.0))
             for i in range(i_min, i_max)}
        isnode = False
    return APDisc(r, m, isnode, apcont.uderivs)


def perturb(formula, eps):
    return stl.perturb(formula, eps)


def ap_cont2d_to_disc(apcont, nodes_coords, elems_nodes, obj_finder, shapes):
    """
    obj_finder :: NodeIndex -> NodeIndex -> [Elem<Dimension>Index, [NodeIndex]]
    shapes :: Dimension x [ShapeFunction]
    """
    region = apcont.A

    size = region[1] - region[0]

    if np.all(isclose(size, 0.0)):
        apf = _ap_cont2d_to_disc_0d_degen
    elif isclose(size[0], 0.0):
        apf = _ap_cont2d_to_disc_1d_x_degen
    elif isclose(size[1], 0.0):
        apf = _ap_cont2d_to_disc_1d_y_degen
    else:
        apf = _ap_cont2d_to_disc_full

    return apf(apcont, nodes_coords, elems_nodes, grid_shape)

def _ap_cont2d_to_disc_0d_degen(apcont, nodes_coords, elems_nodes, obj_finder, shapes):
    node = apcont.A[0]
    n = fem.find_node(node, nodes_coords)
    if apcont.uderivs > 0:
        raise Exception("Derivatives at nodes are not well defined")
    m = {n : (apcont.p(*node), apcont.dp(*node))}
    region_dim = 0

    return APDisc(apcont.r, m, u_comp=apcont.u_comp,
                  uderivs=apcont.uderivs, region_dim=region_dim)

def _ap_cont2d_to_disc_1d_x_degen(apcont, nodes_coords, elems_nodes, obj_finder, shapes):
    ns = [fem.find_node(node, nodes_coords) for node in apcont.A]
    elem_list = obj_finder(ns[0], ns[1], 1)

    m = {n : (apcont.p(PLACEHOLDER), apcont.dp(PLACEHOLDER))
         for n in range(ns[0], ns[1])}
    region_dim = 1

    return APDisc(apcont.r, m, u_comp=apcont.u_comp,
                  uderivs=apcont.uderivs, region_dim=region_dim)

def _ap_cont2d_to_disc_1d_y_degen(apcont, nodes_coords, elems_nodes, obj_finder, shapes):
    ns = [fem.find_node(node, nodes_coords) for node in apcont.A]
    node_list = obj_finder(ns[0], ns[1], 1)

    m = {n : (apcont.p(PLACEHOLDER), apcont.dp(PLACEHOLDER))
         for n in range(ns[0], ns[1], grid_shape[0])}
    region_dim = 1

    return APDisc(apcont.r, m, u_comp=apcont.u_comp,
                  uderivs=apcont.uderivs, region_dim=region_dim)

def _ap_cont2d_to_disc_full(apcont, nodes_coords, elems_nodes, obj_finder, shapes):
    ns = [fem.find_node(node, nodes_coords) for node in apcont.A]
    es = np.aray([fem.find_elem_with_vertex(n, pos, elems_nodes)
                  for n, pos in zip(ns, [0, 3])])
    elem_list = obj_finder(es[0], es[1], 2)

    m = {i * grid_shape[0] + j : (apcont.p(PLACEHOLDER), apcont.dp(PLACEHOLDER))
         for i in range(*(es / grid_shape[0]))
         for j in range(*(es % grid_shape[0]))}
    region_dim = 2

    return APDisc(apcont.r, m, u_comp=apcont.u_comp,
                  uderivs=apcont.uderivs, region_dim=region_dim)


def scale_time(formula, dt):
    formula.bounds = [int(b / dt) for b in formula.bounds]
    for arg in formula.args:
        if arg.op != stl.EXPR:
            scale_time(arg, dt)


class ContModel(object):
    def __init__(self, model):
        self.model = model
        self.tinter = 1

    def getVarByName(self, var_t):
        _, i, t = unlabel(var_t)
        return self.model[t][i]


def csystem_robustness(spec, system, d0, dt):
    # scale_time(spec, dt)
    h = spec.horizon()
    T = dt * h
    t = np.linspace(0, T, h + 1)
    model = ContModel(sys.cont_integrate(system, d0[1:-1], t))

    return stl.robustness(spec, model)


def _Build_f(p, op, isnode, uderivs, elem_len):
    if isnode:
        return lambda vs: (vs[0] - p) * (-1 if op == stl.LE else 1)
    else:
        if uderivs == 0:
            return lambda vs: (.5 * vs[0] + .5 * vs[1] - p) * (-1 if op == stl.LE else 1)
        elif uderivs == 1:
            return lambda vs: ((vs[1] - vs[0]) / elem_len - p) * (-1 if op == stl.LE else 1)

class SysSignal(stl.Signal):
    def __init__(self, index, op, p, dp, isnode, uderivs, xpart=None,
                 fdt_mult=1, bounds=None):
        self.index = index
        self.op = op
        self.p = p
        self.dp = dp
        self.isnode = isnode
        self.uderivs = uderivs
        self.fdt_mult = fdt_mult

        if isnode:
            self.labels = [lambda t: label("d", self.index, t)]
            self.elem_len = 0
        else:
            self.labels = [(lambda t, i=i: label("d", self.index + i, t)) for i in range(2)]
            self.elem_len = xpart[index + 1] - xpart[index]

        self.f = _Build_f(p, op, isnode, uderivs, self.elem_len)
        if bounds is None:
            self.bounds = [-1000, 1000]
        else:
            self.bounds = bounds

    # eps :: index -> isnode -> d/dx mu -> pert
    def perturb(self, eps):
        pert = -eps(self.index, self.isnode, self.dp, self.uderivs)
        self.p = self.p + (pert if self.op == stl.LE else -pert)
        self.f = _Build_f(self.p, self.op, self.isnode, self.uderivs, self.elem_len)

    def signal(self, model, t):
        return super(SysSignal, self).signal(model, t * self.fdt_mult)

    def __str__(self):
        return "{isnode} {uderivs} {index} {op} {p} {dp}".format(
            isnode="d" if self.isnode else "y", uderivs=self.uderivs,
            index=self.index, op="<" if self.op == stl.LE else ">",
            p=self.p, dp=self.dp)

    def __repr__(self):
        return self.__str__()


def expr_parser(xpart=None, fdt_mult=1, bounds=None):
    num = stl.num_parser()

    T_LE = Literal("<")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    isnode = Word(alphas).setParseAction(lambda t: t == "d")
    relation = (T_LE | T_GR).setParseAction(lambda t: stl.LE if t[0] == "<" else stl.GT)
    expr = isnode + integer + integer + relation + num + num
    expr.setParseAction(lambda t: SysSignal(t[2], t[3], t[4], t[5], t[0], t[1],
                                            xpart=xpart, fdt_mult=fdt_mult, bounds=bounds))

    return expr

def stl_parser(xpart=None, fdt_mult=1, bounds=None):
    stl_parser = MatchFirst(stl.stl_parser(expr_parser(xpart, fdt_mult, bounds), True))
    return stl_parser


# TS approach functions

def project_apdisc(apdisc, indices, tpart):
    state_indices = []
    for i in indices:
        if i in apdisc.m:
            if apdisc.r == 1:
                bound_index = bisect_right(tpart, apdisc.m[i]) - 1
                state_indices.append(list(range(bound_index)))
            if apdisc.r == -1:
                bound_index = bisect_left(tpart, apdisc.m[i])
                state_indices.append(list(range(bound_index, len(tpart) - 1)))

    return list(it.product(*state_indices))

def subst_spec_labels(spec, regions):
    res = spec
    for k, v in regions.items():
        if any(isinstance(el, list) or isinstance(el, tuple) for el in v):
            replaced = "(" + " | ".join(["(state = {})".format(state_label(s))
                                          for s in v]) + ")"
        else:
            replaced = "(state = {})".format(state_label(v))
        res = res.replace(k, replaced)
    return res

def project_apdict(apdict, indices, tpart):
    ret = {}
    for key, value in apdict.items():
        ret[key] = project_apdisc(value, indices, tpart)
    return ret


