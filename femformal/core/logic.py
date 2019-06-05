"""
S-STL extension of STL. The module provides classes for
predicates in continuous time (:class:`APCont` and
:class:`APCont2D`) and the individual STL predicates
obtained after discretization (:class:`STLPred`). The discretization is
performed by :func:`sstl_to_stl`.

"""
from __future__ import division, absolute_import, print_function

import itertools as it
from bisect import bisect_left, bisect_right
import logging

import numpy as np
from pyparsing import Word, Literal, MatchFirst, nums
from stlmilp import stl as stl
from enum import Enum

from . import system as sys
from .util import state_label, label, unlabel


logger = logging.getLogger(__name__)

# S-STL data structures

Quantifier = Enum('Quantifier', 'forall exists')

class APCont(object):
    """1D S-STL predicate :math:`Q x \in A : \\frac{d^i u}{d x^i} \\sim p(x)`

    Parameters
    ----------
    A : ndarray
        Spatial domain, given as a closed interval array([a, b])
    r : {'>', '<'}
        Inequality direction.
    p : callable
        The reference profile. Must have signature p(float) -> float
    dp : callable, optional
        The derivative of the reference profile. Must have signature
        dp(float) -> float
    uderivs : int, optional
        The order of the u derivative
    quantifier : :class:`femformal.core.logic.Quantifier`, optional
        The spatial quantifier `Q`

    """

    def __init__(self, A, r, p, dp=None, uderivs=0, quantifier=Quantifier.forall):
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
        self.deriv = -1
        self.quantifier = quantifier


class APCont2D(APCont):
    """2D S-STL predicate :math:`\\forall x \in A : u_j \\sim p(x)`

    If `deriv` is supplied, then the predicate is defined using the
    corresponding directional derivative:
    :math:`\\forall x \in A : \\frac{d u_j}{d x_i} \\sim p(x)`

    Parameters
    ----------
    u_comp : int
        The DOF of u (:math:`j` in the above equation)
    A : ndarray
        Spatial domain, given as the lower left and upper right corners of a
        rectangle
    r : {'>', '<'}
        Inequality direction.
    p : callable
        The reference profile. Must have signature p(float) -> float
    dp : callable
        The derivative of the reference profile. Must have signature
        dp(float) -> float
    deriv : int, optional
        Directional derivative (:math:`i` in the above equation)

    """
    def __init__(self, u_comp, A, r, p, dp, deriv=-1):
        APCont.__init__(self, A, r, p, dp, uderivs=(0 if deriv < 0 else 1))
        self.u_comp = u_comp
        self.deriv = deriv


# S-STL to STL

class _APDisc(object):
    # Intermediate structure between S-STL predicates and STL discretization
    def __init__(self, stlpred_list, quantifier=Quantifier.forall):
        self.stlpred_list = stlpred_list
        self.quantifier = quantifier

    def _quant_str(self):
        if self.quantifier == Quantifier.forall:
            return " & "
        elif self.quantifier == Quantifier.exists:
            return " | "
        else:
            raise ValueError()

    def __str__(self):
        if len(self.stlpred_list) > 1:
            return "({})".format(self._quant_str().join(
                [str(pred) for pred in self.stlpred_list]))
        else:
            return str(self.stlpred_list[0])


class STLPred(object):
    """Regular STL predicate obtained after discretizing an S-STL predicate

    Represents a single `d ~ p` or `y ~ p` predicate.

    The first case has `region_dim = 0` and `d` is
    the `u_comp` component of the value of the `uderivs` derivative at the node
    indexed by `index`.

    The second case has `region_dim > 0` and `y` is the `u_comp` component of
    the value of the `uderivs` derivative at the `query_point` of the element
    indexed by `index`. If the element is multidimensional, 'deriv' specifies
    the direction of the derivative and `uderivs` is restricted to 1.
    Typically an element has one query point (its
    chebyshev center), but elements can implement more query points to decrease
    the conservativeness of the predicate discretization (in particular the
    eta correction).

    Parameters
    ----------
    index : int
        Index of the element or node
    r : {-1, 1}
        Direction of the inequality. -1 means ">", 1 means "<"
    p : float
        Target value in the predicate
    dp : float or array_like
        Derivative or gradient of the target profile at the point
    isnode : bool
        Deprecated parameter, only kept for compatibility. Use region_dim
        instead
    uderivs : int
        Order of the derivative of the field value
    u_comp : int
        Component of the field value
    region_dim : int
        Dimension of the element for which this predicate is defined
    query_point : int
        Index of the query point in the element
    deriv : int, optional
        Directional derivative

    """
    def __init__(self, index, r, p, dp, isnode = True, uderivs = 0, u_comp = 0,
                 region_dim = 0, query_point = 0, deriv = -1):
        self.index = index
        self.r = r
        self.p = p
        self.dp = dp
        self.isnode = region_dim == 0
        self.uderivs = uderivs
        self.u_comp = u_comp
        self.region_dim = region_dim
        self.query_point = query_point
        self.deriv = deriv

    def __str__(self):
        return ("({region_dim} {u_comp} {uderivs} "
                "{index} {query} {op} {p} {dp} {deriv})".format(
            region_dim=self.region_dim,
            u_comp=self.u_comp,
            uderivs=self.uderivs,
            index=self.index,
            op="<" if self.r == 1 else ">",
            p=self.p,
            dp=self.dp,
            query=self.query_point,
            deriv=self.deriv
        ))


def _subst_spec_labels_disc(spec, regions):
    res = spec
    for k, v in regions.items():
        replaced = str(v)
        res = res.replace(k, replaced)
    return res

def _ap_cont_to_disc(apcont, xpart):
    # xpart : [x_i] (list)
    # FIXME TS based approach probably has wrong index assumption
    r = apcont.r
    N1 = len(xpart)
    if apcont.A[0] == apcont.A[1]:
        if apcont.uderivs > 0:
            raise Exception("Derivatives at nodes are not well defined")
        i = min(max(bisect_left(xpart, apcont.A[0]), 0), N1 - 1)
        stlpred_list = [STLPred(
            i, r, apcont.p(xpart[i]), apcont.dp(xpart[i]), isnode=True,
            uderivs=apcont.uderivs, region_dim=0)]
    else:
        if apcont.uderivs > 1:
            raise Exception(
                ("Second and higher order derivatives are 0 for linear "
                "interpolation: uderivs = {}").format(apcont.uderivs))

        i_min = max(bisect_left(xpart, apcont.A[0]), 0)
        i_max = min(bisect_left(xpart, apcont.A[1]), N1 - 1)
        stlpred_list = [
            STLPred(
                i, r,
                apcont.p((xpart[i] + xpart[i+1]) / 2.0),
                apcont.dp((xpart[i] + xpart[i+1]) / 2.0),
                isnode=False, uderivs=apcont.uderivs, region_dim=1,
                deriv=apcont.deriv
            ) for i in range(i_min, i_max)]
    return _APDisc(stlpred_list, apcont.quantifier)

def _ap_cont2d_to_disc(apcont, mesh_):
    region = apcont.A
    build_elem = mesh_.build_elem

    elem_set = mesh_.find_elems_between(region[0], region[1])
    elems = [(e, build_elem(elem_set[e])) for e in elem_set.elems]
    stlpred_list = [
        STLPred(
            e, apcont.r,
            apcont.p(*coords), apcont.dp(*coords),
            isnode=elem_set.dimension == 0, uderivs=apcont.uderivs,
            u_comp=apcont.u_comp, region_dim=elem_set.dimension, query_point=i,
            deriv=apcont.deriv
        ) for e, elem in elems for i, (coords, h) in enumerate(elem.covering())]

    return _APDisc(stlpred_list)

def sstl_to_stl(spec, regions, xpart=None, mesh_=None):
    """Transforms an S-STL spec into an STL spec

    Either `xpart` or `mesh_` must be set.

    Parameters
    ----------
    spec : str
    regions : dict
        Dictionary from labels to S-STL predicates
        (:class:`femformal.core.logic.APCont`)
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh_ : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`

    Returns
    -------
    str
        The STL discretization of the S-STL formula `spec`

    """
    if xpart is not None:
        apc_to_apd = lambda pred: _ap_cont_to_disc(pred, xpart)
    else:
        apc_to_apd = lambda pred: _ap_cont2d_to_disc(pred, mesh_)
    dregions = {label: apc_to_apd(pred) for label, pred in regions.items()}
    dspec = _subst_spec_labels_disc(spec, dregions)
    return dspec


# STL transformations

def perturb(formula, eps):
    """Perturbs the predicates of a formula

    Parameters
    ----------
    formula : :class:`stlmilp.stl.Formula`
    eps : callable
        Computes the perturbation for a given STL predicate (:class:`STLPred`).

    Returns
    -------
    :class:`stlmilp.stl.Formula`
        The perturbed formula

    """
    return stl.perturb(formula, eps)

def scale_time(formula, dt):
    """Transforms a formula in continuous time to discrete time

    Substitutes the time bounds in a :class:`stlmilp.stl.Formula` from
    continuous time to discrete time with time interval `dt`

    Parameters
    ----------
    formula : :class:`stlmilp.stl.Formula`
    dt : float

    Returns
    -------
    None

    """
    formula.bounds = [int(b / dt) for b in formula.bounds]
    for arg in formula.args:
        if arg.op != stl.EXPR:
            scale_time(arg, dt)


# Direct simulation model

class ContModel(object):
    def __init__(self, model):
        self.model = model
        # logger.debug(model)
        self.tinter = 1

    def getVarByName(self, var_t):
        _, i, t = unlabel(var_t)
        return self.model[t][i]


def csystem_robustness(spec, system, d0):
    # scale_time(spec, dt)
    h = max(0, spec.horizon()) + 1
    T = system.dt * (h - 1)
    model = ContModel(sys.integrate(system, d0, T, system.dt))

    return stl.robustness(spec, model)


# MILP simulation model

def _build_f(ap, elem_len):
    p, op, isnode, uderivs = ap.p, ap.r, ap.isnode, ap.uderivs
    if isnode:
        return lambda vs: -(vs[0] - p) * op
    else:
        if uderivs == 0:
            return lambda vs: -(.5 * vs[0] + .5 * vs[1] - p) * op
        elif uderivs == 1:
            return lambda vs: -((vs[1] - vs[0]) / elem_len - p) * op

def _build_f_elem(ap, elem):
    p, op, uderivs = ap.p, ap.r, ap.uderivs
    try:
        logger.debug(p.shape)
    except:
        pass
    if uderivs == 0:
        return lambda vs: - op * (
            elem.interpolate_phys(vs, elem.covering()[ap.query_point][0]) - p)
    else:
        # return lambda vs: - op * (
        #     elem.interpolate_derivatives(
        #         [[vs[i*2 + j] for j in range(2)] for i in range(len(vs) // 2)],
        #         elem.covering()[ap.query_point][0])[ap.u_comp][ap.deriv] - p)

        component = ap.u_comp if ap.u_comp == ap.deriv else 2
        return lambda vs: - op * (
            elem.interpolate_strain(
                vs,
                elem.covering()[ap.query_point][0])[component] - p)

class SysSignal(stl.Signal):
    """Secondary signal from a FEM system

    Parameters
    ----------
    apdisc : :class:`STLPred`
        The predicate defining this secondary signal
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh_ : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`
    fdt_mult : int, optional
        Multiplier of the time interval used in the system discretization. The
        resulting time interval is the one used to discretize in time the
        STL specification. Must be > 1
    bounds : array_like, optional
        [min, max] bounds for the secondary signal. Default is [-1000, 1000]

    """
    def __init__(self, apdisc, xpart=None, fdt_mult=1, bounds=None, mesh_=None):
        self.apdisc = apdisc
        self.fdt_mult = fdt_mult
        self.xpart = xpart
        self.mesh_ = mesh_

        if self.xpart is not None:
            if self.apdisc.isnode:
                self.labels = [lambda t: label("d", self.apdisc.index, t)]
                self.elem_len = 0
            else:
                self.labels = [
                    (lambda t, i=i: label("d", self.apdisc.index + i, t))
                    for i in range(2)]
                self.elem_len = (xpart[self.apdisc.index + 1] -
                                 xpart[self.apdisc.index])
            self.f = _build_f(self.apdisc, self.elem_len)
        else:
            #FIXME dofs?
            if self.apdisc.deriv < 0:
                self.labels = [
                    (lambda t, i=i: label("d", self.apdisc.u_comp + 2 * i, t))
                    for i in mesh_.elem_nodes(
                        self.apdisc.index, self.apdisc.region_dim)]
            else:
                self.labels = [
                    (lambda t, i=i, j=j: label("d", j + 2 * i, t))
                    for i in mesh_.elem_nodes(
                        self.apdisc.index, self.apdisc.region_dim)
                    for j in range(2)
                ]

            self.elem = mesh_.build_elem(
                mesh_.elem_coords(self.apdisc.index, self.apdisc.region_dim))
            self.f = _build_f_elem(self.apdisc, self.elem)

        if bounds is None:
            self.bounds = [-1000, 1000]
        else:
            self.bounds = bounds

    def perturb(self, eps):
        """Perturbs this signal

        Parameters
        ----------
        eps : callable
            Computes the perturbation for a given STL predicate
            (:class:`STLPred`)

        Returns
        -------
        None

        """
        pert = -eps(self.apdisc)
        self.apdisc.p = self.apdisc.p + self.apdisc.r * pert
        if self.xpart is not None:
            self.f = _build_f(self.apdisc, self.elem_len)
        else:
            self.f = _build_f_elem(self.apdisc, self.elem)

    def signal(self, model, t):
        return super(SysSignal, self).signal(model, t * self.fdt_mult)

    def __str__(self):
        return str(self.apdisc)

    def __repr__(self):
        return self.__str__()


# STL parser with S-STL discretized predicates

def _expr_parser(xpart=None, fdt_mult=1, bounds=None, mesh_=None):
    # Must parse the result of STLPred.__str__
    num = stl.num_parser()
    integer = stl.int_parser()

    T_LE = Literal("<")
    T_GR = Literal(">")

    relation = (T_LE | T_GR).setParseAction(lambda t: 1 if t[0] == "<" else -1)
    expr = integer + integer + integer + integer + integer + relation + num + num + integer
    def action(t):
        try:
            signal = SysSignal(
                STLPred(
                    index=t[3],
                    r=t[5],
                    p=t[6],
                    dp=t[7],
                    isnode=False,
                    uderivs=t[2],
                    u_comp=t[1],
                    region_dim=t[0],
                    query_point=t[4],
                    deriv=t[8]
                ),
                fdt_mult=fdt_mult,
                bounds=bounds,
                mesh_=mesh_,
                xpart=xpart
            )
        except Exception as e:
            logger.exception(e)
            raise e

        return signal


    expr.setParseAction(action)

    return expr

def stl_parser(xpart=None, fdt_mult=1, bounds=None, mesh_=None):
    """Builds a parser for STL formulas with :class:`STLPred` predicates

    The returned parser builds a :class:`stlmilp.stl.Formula` with
    :class:`SysSignal` expressions from a specification
    string in the format returned by :func:`sstl_to_stl`.

    All arguments are passed to :class:`SysSignal` constructor and have no
    other effect on the parser.

    Parameters
    ----------
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh_ : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`
    fdt_mult : int, optional
        Multiplier of the time interval used in the system discretization. The
        resulting time interval is the one used to discretize in time the
        STL specification. Must be > 1
    bounds : array_like, optional
        [min, max] bounds for the secondary signal. Default is [-1000, 1000]

    Returns
    -------
    :class:`pyparsing.ParserElement`

    """
    stl_parser = MatchFirst(
        stl.stl_parser(_expr_parser(xpart, fdt_mult, bounds, mesh_), True))
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


