import stlmilp.milp_util as milp
import stlmilp.stl as stl
import femformal.system as sys
import numpy as np
from collections import deque
from pyparsing import Word, Literal, MatchFirst, nums, alphas

import logging
logger = logging.getLogger('FEMMILP')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s %(module)s:%(lineno)d:%(funcName)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logger.setLevel(logging.DEBUG)

def rh_system_sat(system, d0, N, spec, thunk=None):
    # print spec
    hd = max(0, spec.horizon())
    H = hd
    dcur = d0
    # dhist = deque([xcur])

    for j in range(N - 1):
        m = milp.create_milp("rhc_system")
        logger.debug("Adding affine system constraints")
        if isinstance(system, sys.System):
            d = milp.add_affsys_constr_x0(
                m, "d", system.A, system.b, dcur, H, None)
        elif isinstance(system, sys.SOSystem):
            d = milp.add_newmark_constr_x0(
                m, "d", system.M, system.K, system.F, dcur, thunk['dt'], H, None)
        else:
            raise Exception(
                "Not implemented for this class of system: {}".format(
                    system.__class__.__name__))
        logger.debug("Adding STL constraints")
        fvar, vbds = milp.add_stl_constr(m, "spec", spec)
        # m.addConstr(fvar >= 0)
        m.params.outputflag = 0
        m.update()
        # if j == 0:
        #     m.write("out.lp")
        logger.debug("Optimizing")
        m.optimize()
        logger.debug("Finished optimizing")
        if j == 0:
            for v in m.getVars():
                print v

        if m.status != milp.GRB.status.OPTIMAL:
            logger.warning("MILP returned status: {}".format(m.status))
            return False
        else:
            d_opt = m.getAttr("x", d)
            if isinstance(system, sys.System):
                dcur = [d_opt[milp.label("d", i, 1)] for i in range(system.n + 2)]
            elif isinstance(system, sys.SOSystem):
                dcur = [d_opt[milp.label("d", i, 1)] for i in range(system.n)]
            # dhist.append(dcur)
            # if j > hd:
            #     dhist.popleft()
        # print [d_opt[milp.label("d", i, H - 1)] for i in range(system.n)]

    return fvar.getAttr("x")

def rh_system_sat_set(system, pset, f, xpart, N, spec, thunk=None):
    # print spec
    hd = max(0, spec.horizon())
    H = hd

    m = milp.create_milp("rhc_system")
    logger.debug("Adding affine system constraints")
    if isinstance(system, sys.System):
        d = milp.add_affsys_constr_x0_set(
            m, "d", system.A, system.b, H, xpart, pset, f)
    elif isinstance(system, sys.SOSystem):
        d = milp.add_newmark_constr_x0_set(
            m, "d", system.M, system.K, system.F, xpart, pset, f, thunk['dt'], H)
    else:
        raise Exception(
            "Not implemented for this class of system: {}".format(
                system.__class__.__name__))
    logger.debug("Adding STL constraints")
    fvar, vbds = milp.add_stl_constr(m, "spec", spec)
    fvar.setAttr("obj", 1.0)
    m.params.outputflag = 0
    m.update()
    # if j == 0:
    #     m.write("out.lp")
    logger.debug("Optimizing")
    m.optimize()
    logger.debug("Finished optimizing")
    # if j == 0:
    #     for v in m.getVars():
    #         print v

    if m.status != milp.GRB.status.OPTIMAL:
        logger.warning("MILP returned status: {}".format(m.status))
        return False
    else:
        return fvar.getAttr("x")

class ContModel(object):
    def __init__(self, model):
        self.model = model
        self.tinter = 1

    def getVarByName(self, var_t):
        _, i, t = milp.unlabel(var_t)
        return self.model[t][i]

def csystem_robustness(spec, system, d0, dt):
    scale_time(spec, dt)
    h = spec.horizon()
    T = dt * h
    t = np.linspace(0, T, h + 1)
    model = ContModel(sys.cont_integrate(system, d0[1:-1], t))

    return milp.robustness(spec, model)

def _Build_f(p, op, isnode):
    if isnode:
        return lambda vs: (vs[0] - p) * (-1 if op == stl.LE else 1)
    else:
        return lambda vs: (.5 * vs[0] + .5 * vs[1] - p) * (-1 if op == stl.LE else 1)

class SysSignal(stl.Signal):
    def __init__(self, index, op, p, dp, isnode):
        self.index = index
        self.op = op
        self.p = p
        self.dp = dp
        self.isnode = isnode

        if isnode:
            self.labels = [lambda t: milp.label("d", self.index, t)]
        else:
            self.labels = [(lambda t, i=i: milp.label("d", self.index + i, t)) for i in range(2)]

        self.f = _Build_f(p, op, isnode)
        self.bounds = [-10, 10] #FIXME bounds

    # eps :: index -> isnode -> d/dx mu -> pert
    def perturb(self, eps):
        pert = -eps(self.index, self.isnode, self.dp)
        self.p = self.p + (pert if self.op == stl.LE else -pert)
        self.f = _Build_f(self.p, self.op, self.isnode)

    def __str__(self):
        return "{} {} {} {} {}".format("d" if self.isnode else "y", self.index,
                                    "<" if self.op == stl.LE else ">", self.p, self.dp)

    def __repr__(self):
        return self.__str__()

def scale_time(formula, dt):
    formula.bounds = [int(b / dt) for b in formula.bounds]
    for arg in formula.args:
        if arg.op != stl.EXPR:
            scale_time(arg, dt)

def perturb(formula, eps):
    return stl.perturb(formula, eps)

def expr_parser():
    num = stl.num_parser()

    T_LE = Literal("<")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    isnode = Word(alphas).setParseAction(lambda t: t == "d")
    relation = (T_LE | T_GR).setParseAction(lambda t: stl.LE if t[0] == "<" else stl.GT)
    expr = isnode + integer + relation + num + num
    expr.setParseAction(lambda t: SysSignal(t[1], t[2], t[3], t[4], t[0]))

    return expr

def stl_parser():
    stl_parser = MatchFirst(stl.stl_parser(expr_parser(), True))
    return stl_parser
