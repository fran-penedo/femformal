import core.system as sys
import gurobipy as g
from util import label

import logging
logger = logging.getLogger('FEMFORMAL')

def add_sys_constr_x0(m, l, system, x0, N, xhist=None):
    if isinstance(system, sys.System):
        x = add_affsys_constr_x0(m, l, system, x0, N, xhist)
    elif isinstance(system, sys.SOSystem):
        x = add_newmark_constr_x0(m, l, system, x0, N, xhist)
    else:
        raise Exception(
            "Not implemented for this class of system: {}".format(
                system.__class__.__name__))
    return x

def add_sys_constr_x0_set(m, l, system, pset, f, N):
    if isinstance(system, sys.System):
        x = add_affsys_constr_x0_set(m, l, system, pset, f, N)
    elif isinstance(system, sys.SOSystem):
        x = add_newmark_constr_x0_set(m, l, system, pset, f, N)
    else:
        raise Exception(
            "Not implemented for this class of system: {}".format(
                system.__class__.__name__))
    return x


def add_affsys_constr_x0(m, l, system, x0, N, xhist=None):
    x = add_affsys_constr(m, l, system, N, xhist)
    for j in range(1, N):
        for i in [0, system.n + 1]:
            m.addConstr(x[label(l, i, j)] == x0[i])
    for i in range(system.n + 2):
        m.addConstr(x[label(l, i, 0)] == x0[i])
    return x

def add_affsys_constr_x0_set(m, l, system, pset, f, N):
    x = add_affsys_constr(m, l, system, N, None)
    xpart = system.xpart

    # p0 = m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name='p0')
    # p1 = m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name='p1')
    p = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('p', i, 0))
          for i in range(pset.shape[1] - 1)]
    m.update()

    for i in range(pset.shape[0]):
        m.addConstr(g.quicksum(
            pset[i][j] * p[j] for j in range(pset.shape[1] - 1)) <= pset[i][-1])
    # for i in range(pset.shape[0]):
    #     m.addConstr(p0 * pset[i][0] + p1 * pset[i][1] <= pset[i][2])

    for i in range(len(xpart)):
        m.addConstr(x[label(l, i, 0)] == f(xpart[i], p))
    for j in range(1, N):
        for i in [0, system.n + 1]:
            m.addConstr(x[label(l, i, j)] == f(xpart[i], p))

    return x

def add_affsys_constr(m, l, system, N, xhist=None):
    if xhist is None:
        xhist = []
    A = system.A
    b = system.b

    # Decision variables
    logger.debug("Adding decision variables")
    x = {}
    for i in range(A.shape[0] + 2):
        for j in range(-len(xhist), N):
            labelx = label(l, i, j)
            x[labelx] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelx)
    m.update()

    # Dynamics
    logger.debug("Adding dynamics")
    for i in range(1, A.shape[0] + 1):
        logger.debug("Adding row {}".format(i))
        for j in range(-len(xhist), N):
            if j < 0:
                m.addConstr(x[label(l, i, j)] == xhist[len(xhist) + j][i])
            elif j > 0:
                m.addConstr(x[label(l, i, j)] ==
                            g.quicksum(A[i-1][k] * x[label(l, k + 1, j - 1)]
                                       for k in range(A.shape[0])) + b[i-1])
    m.update()

    return x


def add_newmark_constr_x0_set(m, l, system, pset, f, N):
    x = add_newmark_constr(m, l, system, N, None)

    xpart = system.xpart
    dset, vset, fset = pset
    fd, fv, ff = f

    pd = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pd', i, 0))
          for i in range(dset.shape[1] - 1)]
    pv = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pv', i, 0))
          for i in range(vset.shape[1] - 1)]
    pf = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pf', i, 0))
          for i in range(fset.shape[1] - 1)]

    for i in range(dset.shape[0]):
        m.addConstr(g.quicksum(
            pd[j] * dset[i][j] for j in range(dset.shape[1] - 1)) <= dset[i][-1])
    for i in range(vset.shape[0]):
        m.addConstr(g.quicksum(
            pv[j] * vset[i][j] for j in range(vset.shape[1] - 1)) <= vset[i][-1])
    for i in range(fset.shape[0]):
        m.addConstr(g.quicksum(
            pf[j] * fset[i][j] for j in range(fset.shape[1] - 1)) <= fset[i][-1])

    for i in range(len(xpart)):
        m.addConstr(x[label(l, i, 0)] == fd(xpart[i], pd))
        m.addConstr(x[label('d' + l, i, 0)] == fv(xpart[i], pv))
        m.addConstr(x[label('f', i, 0)] == ff(xpart[i], pf))

    return x

def add_newmark_constr_x0(m, l, system, x0, N, xhist=None):
    x = add_newmark_constr(m, l, system, N, xhist)
    d0, v0 = x0
    # for j in range(1, N):
    #     for i in [0, M.shape[0] + 1]:
    #         m.addConstr(x[label(l, i, j)] == d0[i])
    #         m.addConstr(x[label('d' + l, i, j)] == v0[i])
    for i in range(system.n):
        m.addConstr(x[label(l, i, 0)] == d0[i])
        m.addConstr(x[label('d' + l, i, 0)] == v0[i])
    return x

def add_newmark_constr(m, l, system, N, xhist=None):
    if xhist is None:
        xhist = []
    M, K, F, dt = system.M, system.K, system.F, system.dt

    # Decision variables
    logger.debug("Adding decision variables")
    x = {}
    for i in range(M.shape[0]):
        labelf = label('f', i, 0)
        x[labelf] = m.addVar(
            obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelf)
        for j in range(-len(xhist), N):
            labelx = label(l, i, j)
            x[labelx] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelx)
            labelv = label('d' + l, i, j)
            x[labelv] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelv)
            labeltv = label('td' + l, i, j)
            x[labeltv] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labeltv)
            labela = label('dd' + l, i, j)
            x[labela] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labela)

    m.update()

    # Dynamics
    logger.debug("Adding dynamics")
    for i in range(M.shape[0]):
        logger.debug("Adding row {}".format(i))
        for j in range(-len(xhist), N):
            if j < 0:
                m.addConstr(x[label(l, i, j)] == xhist[len(xhist) + j][i])
            elif j == 0:
                # M a = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('dd' + l, i, j)] == 0)
                else:
                    m.addConstr(g.quicksum(M[i, k] * x[label('dd' + l, k, j)]
                                        for k in range(M.shape[0])) ==
                                x[label('f', i, 0)] + F[i] -
                                g.quicksum(K[i, k] * x[label(l, k, j)]
                                                for k in range(K.shape[0])))
            elif j > 0:
                # tv = v + .5 * dt * a
                m.addConstr(x[label('td' + l, i, j)] == x[label('d' + l, i, j - 1)] +
                            .5 * dt * x[label('dd' + l, i, j - 1)])
                # d = d + dt * tv
                m.addConstr(x[label(l, i, j)] == x[label(l, i, j - 1)] +
                            dt * x[label('td' + l, i, j)])
                # M a = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('dd' + l, i, j)] == 0)
                else:
                    m.addConstr(g.quicksum(M[i, k] * x[label('dd' + l, k, j)]
                                        for k in range(M.shape[0])) ==
                                x[label('f', i, 0)] + F[i] -
                                g.quicksum(K[i, k] * x[label(l, k, j)]
                                                for k in range(K.shape[0])))
                # v = tv + .5 * dt * a
                m.addConstr(x[label('d' + l, i, j)] == x[label('td' + l, i, j)] +
                            .5 * dt * x[label('dd' + l, i, j)])
    m.update()

    return x
