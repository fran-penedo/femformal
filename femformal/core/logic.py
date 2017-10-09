import itertools as it
from bisect import bisect_left, bisect_right
import logging

import numpy as np
from pyparsing import Word, Literal, MatchFirst, nums, alphas
from stlmilp import stl as stl

from . import system as sys
from .util import state_label, label, unlabel


logger = logging.getLogger(__name__)

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
        self.u_comp = u_comp


class APDisc(object):
    def __init__(self, r, m, isnode = True, uderivs = 0, region_dim = 0, u_comp = 0):
        # r == 1: f < p, r == -1: f > p
        self.r = r
        self.region_dim = region_dim
        self.isnode = region_dim == 0

        # m : i -> (p((x_i + x_{i+1})/2) if not isnode else i -> p(x_i),
        # dp(.....))
        self.m = m
        self.uderivs = uderivs
        self.u_comp = u_comp

    def __str__(self):
        return "({})".format(" & ".join(
                ["({region_dim} {u_comp} {uderivs} {index} {op} {p} {dp})".format(
                    region_dim=self.region_dim,
                    u_comp=self.u_comp,
                    uderivs=self.uderivs,
                    index=i,
                    op="<" if self.r == 1 else ">",
                    p=p,
                    dp=dp
                ) for (i, (p, dp)) in self.m.items()]))

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
        m = {i: (apcont.p(xpart[i]), apcont.dp(xpart[i]))}
        isnode = True
        region_dim = 0
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
        region_dim = 1
    return APDisc(r, m, isnode, apcont.uderivs, region_dim=region_dim)

def ap_cont2d_to_disc(apcont, mesh_, build_elem):
    region = apcont.A

    elem_set = mesh_.find_elems_between(region[0], region[1])
    elems = [(e, build_elem(elem_set[e])) for e in elem_set.elems]
    m = {e : (elem.interpolate([apcont.p(*coord) for coord in elem.coords],
                               [0, 0]),
              elem.interpolate([apcont.dp(*coord) for coord in elem.coords],
                               [0, 0]))
         for e, elem in elems}

    return APDisc(
        apcont.r, m, u_comp=apcont.u_comp, uderivs=apcont.uderivs,
        region_dim=elem_set.dimension)


def perturb(formula, eps):
    return stl.perturb(formula, eps)

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


def _build_f(p, op, isnode, uderivs, elem_len):
    if isnode:
        return lambda vs: (vs[0] - p) * (-1 if op == stl.LE else 1)
    else:
        if uderivs == 0:
            return lambda vs: (.5 * vs[0] + .5 * vs[1] - p) * (-1 if op == stl.LE else 1)
        elif uderivs == 1:
            return lambda vs: ((vs[1] - vs[0]) / elem_len - p) * (-1 if op == stl.LE else 1)

def _build_f_elem(p, op, uderivs, elem):
    try:
        logger.debug(p.shape)
    except:
        pass
    if uderivs == 0:
        return lambda vs: (elem.interpolate_phys(vs, elem.chebyshev_center())
                           - p) * (-1 if op == stl.LE else 1)
    else:
        #FIXME think about this when implementing derivatives
        return lambda vs: (elem.interpolate_derivatives(vs, [0 for i in range(elem.dimension)])
                           - p) * (-1 if op == stl.LE else 1)

class SysSignal(stl.Signal):
    def __init__(self, index, op, p, dp, isnode, uderivs, u_comp=0, region_dim=0,
                 xpart=None, fdt_mult=1, bounds=None, mesh_=None, build_elem=None):
        self.index = index
        self.op = op
        self.p = p
        self.dp = dp
        self.region_dim = region_dim
        self.isnode = region_dim == 0
        self.uderivs = uderivs
        self.fdt_mult = fdt_mult
        self.u_comp = u_comp
        self.xpart = xpart
        self.mesh_ = mesh_
        self.build_elem = build_elem

        if self.xpart is not None:
            if self.isnode:
                self.labels = [lambda t: label("d", self.index, t)]
                self.elem_len = 0
            else:
                self.labels = [(lambda t, i=i: label("d", self.index + i, t)) for i in range(2)]
                self.elem_len = xpart[index + 1] - xpart[index]
            self.f = _build_f(self.p, self.op, self.isnode, self.uderivs, self.elem_len)
        else:
            #FIXME dofs?
            self.labels = [(lambda t, i=i: label("d", self.u_comp + 2 * i, t))
                            for i in mesh_.elem_nodes(self.index, self.region_dim)]
            self.elem = build_elem(mesh_.elem_coords(self.index, self.region_dim))
            self.f = _build_f_elem(self.p, self.op, self.uderivs, self.elem)

        if bounds is None:
            self.bounds = [-1000, 1000]
        else:
            self.bounds = bounds

    # eps :: index -> isnode -> d/dx mu -> pert
    def perturb(self, eps):
        pert = -eps(self.index, self.isnode, self.dp, self.uderivs, self.u_comp, self.region_dim)
        # try:
        #     logger.debug(pert.shape)
        # except:
        #     pass
        self.p = self.p + (pert if self.op == stl.LE else -pert)
        if self.xpart is not None:
            self.f = _build_f(self.p, self.op, self.isnode, self.uderivs, self.elem_len)
        else:
            self.f = _build_f_elem(self.p, self.op, self.uderivs, self.elem)

    def signal(self, model, t):
        return super(SysSignal, self).signal(model, t * self.fdt_mult)

    def __str__(self):
        return "{region_dim} {u_comp} {uderivs} {index} {op} {p} {dp}".format(
            region_dim=self.region_dim,
            uderivs=self.uderivs,
            index=self.index,
            op="<" if self.op ==stl.LE else ">",
            p=self.p,
            dp=self.dp,
            u_comp=self.u_comp,
        )

    def __repr__(self):
        return self.__str__()


def expr_parser(xpart=None, fdt_mult=1, bounds=None, mesh_=None, build_elem=None):
    num = stl.num_parser()

    T_LE = Literal("<")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    relation = (T_LE | T_GR).setParseAction(lambda t: stl.LE if t[0] == "<" else stl.GT)
    expr = integer + integer + integer + integer + relation + num + num
    expr.setParseAction(lambda t: SysSignal(
        index=t[3],
        op=t[4],
        p=t[5],
        dp=t[6],
        isnode=False,
        uderivs=t[2],
        u_comp=t[1],
        region_dim=t[0],
        fdt_mult=fdt_mult,
        bounds=bounds,
        mesh_=mesh_,
        build_elem=build_elem,
        xpart=xpart))

    return expr

def stl_parser(xpart=None, fdt_mult=1, bounds=None, mesh_=None, build_elem=None):
    stl_parser = MatchFirst(
        stl.stl_parser(expr_parser(xpart, fdt_mult, bounds, mesh_, build_elem), True))
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


