import stlmilp.stl as stl
import core.system as sys
import numpy as np
import itertools as it
from bisect import bisect_left, bisect_right
from .util import state_label, label, unlabel
from pyparsing import Word, Literal, MatchFirst, nums, alphas

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

class APDisc(object):
    def __init__(self, r, m, isnode, uderivs = 0):
        # r == 1: f < p, r == -1: f > p
        self.r = r
        self.isnode = isnode
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


def _Build_f(p, op, isnode, uderivs):
    if isnode:
        return lambda vs: (vs[0] - p) * (-1 if op == stl.LE else 1)
    else:
        if uderivs == 0:
            return lambda vs: (.5 * vs[0] + .5 * vs[1] - p) * (-1 if op == stl.LE else 1)
        elif uderivs == 1:
            return lambda vs: (vs[1] - vs[0] - p) * (-1 if op == stl.LE else 1)

class SysSignal(stl.Signal):
    def __init__(self, index, op, p, dp, isnode, uderivs, fdt_mult=1):
        self.index = index
        self.op = op
        self.p = p
        self.dp = dp
        self.isnode = isnode
        self.uderivs = uderivs
        self.fdt_mult = fdt_mult

        if isnode:
            self.labels = [lambda t: label("d", self.index, t)]
        else:
            self.labels = [(lambda t, i=i: label("d", self.index + i, t)) for i in range(2)]

        self.f = _Build_f(p, op, isnode, uderivs)
        self.bounds = [-10, 10] #FIXME bounds

    # eps :: index -> isnode -> d/dx mu -> pert
    def perturb(self, eps):
        pert = -eps(self.index, self.isnode, self.dp)
        self.p = self.p + (pert if self.op == stl.LE else -pert)
        self.f = _Build_f(self.p, self.op, self.isnode, self.uderivs)

    def signal(self, model, t):
        return super(SysSignal, self).signal(model, t * self.fdt_mult)

    def __str__(self):
        return "{isnode} {uderivs} {index} {op} {p} {dp}".format(
            isnode="d" if self.isnode else "y", uderivs=self.uderivs,
            index=self.index, op="<" if self.op == stl.LE else ">",
            p=self.p, dp=self.dp)

    def __repr__(self):
        return self.__str__()


def expr_parser(fdt_mult=1):
    num = stl.num_parser()

    T_LE = Literal("<")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    isnode = Word(alphas).setParseAction(lambda t: t == "d")
    relation = (T_LE | T_GR).setParseAction(lambda t: stl.LE if t[0] == "<" else stl.GT)
    expr = isnode + integer + integer + relation + num + num
    expr.setParseAction(lambda t: SysSignal(t[2], t[3], t[4], t[5], t[0], t[1],
                                            fdt_mult=fdt_mult))

    return expr

def stl_parser(fdt_mult=1):
    stl_parser = MatchFirst(stl.stl_parser(expr_parser(fdt_mult), True))
    return stl_parser


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


