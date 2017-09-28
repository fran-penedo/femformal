import logging

import gurobipy as g
import numpy as np
from stlmilp import milp_util as milp_util

from . import system as sys
from .util import label


logger = logging.getLogger(__name__)

def add_sys_constr_x0(m, l, system, x0, N, xhist=None):
    if isinstance(system, sys.System):
        x = add_affsys_constr_x0(m, l, system, x0, N, xhist)
    elif isinstance(system, sys.FOSystem):
        x = add_trapez_constr_x0(m, l, system, x0, N, xhist)
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
    elif isinstance(system, sys.FOSystem):
        x = add_trapez_constr_x0_set(m, l, system, pset, f, N)
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


def add_trapez_constr_x0_set(m, l, system, pset, f, N):
    x = add_trapez_constr(m, l, system, N, None)

    xpart = system.xpart
    mesh = system.mesh
    dset, fset = pset
    fd, ff = f

    logger.debug("Adding parameter constraints")
    pd = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pd', i, 0))
          for i in range(dset.shape[1] - 1)]
    pf = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pf', i, 0))
          for i in range(fset.shape[1] - 1)]

    m.update()

    for i in range(dset.shape[0]):
        m.addConstr(g.quicksum(
            pd[j] * dset[i][j] for j in range(dset.shape[1] - 1)) <= dset[i][-1])
    for i in range(fset.shape[0]):
        m.addConstr(g.quicksum(
            pf[j] * fset[i][j] for j in range(fset.shape[1] - 1)) <= fset[i][-1])

    logger.debug("Adding initial and boundary conditions")
    if xpart is not None:
        for i in range(len(xpart)):
            m.addConstr(x[label(l, i, 0)] == fd(xpart[i], pd))
            for j in range(N):
                m.addConstr(x[label('f', i, j)] == ff(j * system.dt, pf, xpart[i]))
    else:
        raise NotImplementedError()
        # for i in range(mesh.nnodes):
        #     logger.debug("Adding IC and BC for node {}".format(i))
        #     for dof in range(2):
        #         m.addConstr(x[label(l, i * 2 + dof, 0)] == fd(mesh.nodes_coords[i], pd))
        #         m.addConstr(x[label('d' + l, i * 2 + dof, 0)] == fv(mesh.nodes_coords[i], pv))
        #         for j in range(N):
        #             m.addConstr(x[label('f', i * 2 + dof, j)] == ff(j * system.dt, pf, i * 2 + dof))

    return x

def add_trapez_constr_x0(m, l, system, x0, N, xhist=None):
    x = add_trapez_constr(m, l, system, N, xhist)
    d0 = x0
    for i in range(system.n):
        m.addConstr(x[label(l, i, 0)] == d0[i])
        for j in range(N):
            m.addConstr(x[label('f', i, j)] == 0)
    return x

def add_trapez_constr(m, l, system, N, xhist=None):
    if xhist is None:
        xhist = []
    M, F, dt = system.M, system.F, system.dt

    # Decision variables
    x = add_fosys_variables(m, l, M.shape[0], N, len(xhist))

    if hasattr(system, "K_global"):
        deltas = add_hybrid_K_deltas(m, system.K_els(), x, l, N, system.bigN)
        el_int_forces = [elements_int_force(
            m, l, system.K_els(), x, deltas, time, system.bigN)
            for time in range(N)]
    else:
        el_int_forces = None

    # Dynamics
    logger.debug("Adding dynamics")
    for i in range(M.shape[0]):
        logger.debug("Adding row {}".format(i))
        for j in range(-len(xhist), N):
            if j < 0:
                m.addConstr(x[label(l, i, j)] == xhist[len(xhist) + j][i])
            elif j == 0:
                # M v = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('d' + l, i, j)] == 0)
                else:
                    m.addConstr(g.quicksum(M[i, k] * x[label('d' + l, k, j)]
                                        for k in range(M.shape[0])) ==
                                x[label('f', i, j)] + F[i] -
                                int_force(system, x, i, j, l, el_int_forces))
            elif j > 0:
                # d = d + dt * v
                m.addConstr(x[label(l, i, j)] == x[label(l, i, j - 1)] +
                            dt * x[label('d' + l, i, j - 1)])
                # M v = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('d' + l, i, j)] == 0)
                else:
                    m.addConstr(g.quicksum(M[i, k] * x[label('d' + l, k, j)]
                                        for k in range(M.shape[0])) ==
                                x[label('f', i, j)] + F[i] -
                                int_force(system, x, i, j, l, el_int_forces))
    m.update()

    return x


def add_newmark_constr_x0_set(m, l, system, pset, f, N):
    x = add_newmark_constr(m, l, system, N, None)

    xpart = system.xpart
    mesh = system.mesh
    dset, vset, fset = pset
    fd, fv, ff = f

    logger.debug("Adding parameter constraints")
    pd = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pd', i, 0))
          for i in range(dset.shape[1] - 1)]
    pv = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pv', i, 0))
          for i in range(vset.shape[1] - 1)]
    pf = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pf', i, 0))
          for i in range(fset.shape[1] - 1)]

    m.update()

    for i in range(dset.shape[0]):
        m.addConstr(g.quicksum(
            pd[j] * dset[i][j] for j in range(dset.shape[1] - 1)) <= dset[i][-1])
    for i in range(vset.shape[0]):
        m.addConstr(g.quicksum(
            pv[j] * vset[i][j] for j in range(vset.shape[1] - 1)) <= vset[i][-1])
    for i in range(fset.shape[0]):
        m.addConstr(g.quicksum(
            pf[j] * fset[i][j] for j in range(fset.shape[1] - 1)) <= fset[i][-1])

    logger.debug("Adding initial and boundary conditions")
    if xpart is not None:
        for i in range(len(xpart)):
            m.addConstr(x[label(l, i, 0)] == fd(xpart[i], pd))
            m.addConstr(x[label('d' + l, i, 0)] == fv(xpart[i], pv))
            for j in range(N):
                m.addConstr(x[label('f', i, j)] == ff(j * system.dt, pf, xpart[i]))
    else:
        for i in range(mesh.nnodes):
            logger.debug("Adding IC and BC for node {}".format(i))
            for dof in range(2):
                m.addConstr(x[label(l, i * 2 + dof, 0)] == fd(mesh.nodes_coords[i], pd))
                m.addConstr(x[label('d' + l, i * 2 + dof, 0)] == fv(mesh.nodes_coords[i], pv))
                for j in range(N):
                    m.addConstr(x[label('f', i * 2 + dof, j)] == ff(j * system.dt, pf, i * 2 + dof))

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
        for j in range(N):
            m.addConstr(x[label('f', i, j)] == 0)
    return x

def add_newmark_constr(m, l, system, N, xhist=None):
    if xhist is None:
        xhist = []
    M, F, dt = system.M, system.F, system.dt

    # Decision variables
    x = add_sosys_variables(m, l, M.shape[0], N, len(xhist))

    if hasattr(system, "K_global"):
        deltas = add_hybrid_K_deltas(m, system.K_els(), x, l, N, system.bigN)
        el_int_forces = [elements_int_force(
            m, l, system.K_els(), x, deltas, time, system.bigN)
            for time in range(N)]
    else:
        el_int_forces = None

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
                                x[label('f', i, j)] + F[i] -
                                int_force(system, x, i, j, l, el_int_forces))
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
                                x[label('f', i, j)] + F[i] -
                                int_force(system, x, i, j, l, el_int_forces))
                # v = tv + .5 * dt * a
                m.addConstr(x[label('d' + l, i, j)] == x[label('td' + l, i, j)] +
                            .5 * dt * x[label('dd' + l, i, j)])
    m.update()

    return x

def elements_int_force(m, l, kels, x, deltas, time, bigN):
    fs = []
    for n_el in range(len(kels)):
        el_fs = []
        kel = kels[n_el]
        n = kel.values[0].shape[0]
        for i in range(n):
            v1 = g.quicksum(
                kel.values[0][i][j] * x[label(l, n_el * (n - 1) + j, time)]
                for j in range(n))
            v2 = g.quicksum(
                kel.values[1][i][j] * x[label(l, n_el * (n - 1) + j, time)]
                for j in range(n))
            bigNN = bigN * np.max(np.abs(kel.values))
            f = milp_util.add_binary_switch(
                m, "elintf_{}_{}_{}".format(n_el, i, time), v1, v2,
                deltas[time][n_el], bigNN)
            el_fs.append(f)
        fs.append(el_fs)

    return fs

def int_force(system, x, row, time, l, el_int_forces=None):
    if el_int_forces is None:
        f = g.quicksum(system.K[row, k] * x[label(l, k, time)]
                        for k in range(system.K.shape[0]))
    else:
        if row == 0:
            f = el_int_forces[time][0][0]
        elif row == system.M.shape[0] - 1:
            f = el_int_forces[time][-1][1]
        else:
            f = el_int_forces[time][row - 1][1] + el_int_forces[time][row][0]

    return f


def add_sosys_variables(m, l, nvars, horlen, histlen):
    logger.debug("Adding decision variables")
    x = {}
    for i in range(nvars):
        for j in range(-histlen, horlen):
            labelf = label('f', i, j)
            x[labelf] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelf)
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
    return x

def add_fosys_variables(m, l, nvars, horlen, histlen):
    logger.debug("Adding decision variables")
    x = {}
    for i in range(nvars):
        for j in range(-histlen, horlen):
            labelf = label('f', i, j)
            x[labelf] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelf)
            labelx = label(l, i, j)
            x[labelx] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelx)
            labelv = label('d' + l, i, j)
            x[labelv] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelv)
    m.update()
    return x

def add_hybrid_K_deltas(m, Ks, x, l, N, bigN):
    deltas = []
    for t in range(N):
        deltast = []
        for i in range(len(Ks)):
            delta = milp_util.add_set_flag(m, "K_{}_{}_delta".format(t, i),
                                   [x[label(l, a, t)] for a in range(i, i+2)],
                                   Ks[i].invariants[0][0], Ks[i].invariants[0][1],
                                           bigN)
            deltast.append(delta)
        deltas.append(deltast)

    return deltas

def get_trajectory_from_model(m, l, T, system):
    return np.array([
        [m.getVarByName(label(l, i, j)).x for i in range(system.n)]
        for j in range(T)])
