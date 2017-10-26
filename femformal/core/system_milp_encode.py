"""
MILP encodings of system trajectories
"""
from __future__ import division, absolute_import, print_function

import logging

import gurobipy as g
import numpy as np
from scipy.sparse import linalg as spla, csc_matrix
from stlmilp import milp_util as milp_util

from . import system as sys
from .util import label


logger = logging.getLogger(__name__)

def add_sys_constr_x0(m, l, system, x0, N, xhist=None):
    """Adds a MILP representation of the IVP to a `gurobi` model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : any `System` class in :mod:`femformal.core.system`
    x0 : array_like, shape ([system order], state dimension, [dofs])
        Initial value.
    N : int
        Number of iterations for the desired trajectory, including the initial
        value

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    """
    if isinstance(system, sys.System):
        x = add_affsys_constr_x0(m, l, system, x0, N, xhist)
    elif isinstance(system, sys.FOSystem):
        x = add_trapez_constr_x0(m, l, system, x0, N, xhist)
    elif isinstance(system, sys.SOSystem):
        if isinstance(system, sys.HybridSOSystem):
            x = add_central_diff_constr_x0(m, l, system, x0, N, xhist)
        else:
            x = add_newmark_constr_x0(m, l, system, x0, N, xhist)
    else:
        raise Exception(
            "Not implemented for this class of system: {}".format(
                system.__class__.__name__))
    return x

def add_sys_constr_x0_set(m, l, system, pset, f, N):
    """Adds the parameterized system trajectory to a `gurobi` model

    See the documentation of each `add_x_constr_x0_set` for more details on
    the parameters `pset` and `f`.

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : any `System` class in :mod:`femformal.core.system`
    pset : list of array_like
        Each element of the list is the H-representation of a polytope in
        which some parameters of the system is contained
    f : list of callable
        Each element is a function parameterized by the corresponding parameters
        defined by `pset`
    N : int
        Number of iterations for the desired trajectory, including the initial
        value

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    """
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
    x = _add_affsys_constr(m, l, system, N, xhist)
    for j in range(1, N):
        for i in [0, system.n + 1]:
            m.addConstr(x[label(l, i, j)] == x0[i])
    for i in range(system.n + 2):
        m.addConstr(x[label(l, i, 0)] == x0[i])
    return x

def add_affsys_constr_x0_set(m, l, system, pset, f, N):
    x = _add_affsys_constr(m, l, system, N, None)
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

def _add_affsys_constr(m, l, system, N, xhist=None):
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
    """Adds the parameterized FO system trajectory to a `gurobi` model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : :class:`femformal.core.system.FOSystem`
    pset : list of array_like, (pd, pf)
        Polytopes of the initial value and nodal foce parameters.
    f : list of callable, (fd, ff)
        Parameterized initial value and nodal force functions. `fd` must map
        a point in the domain and the list of parameters to the initial value at
        that point. `ff` must map a time, the list of parameters and a point
        of the domain to the nodal force at that point and time. The parameters
        are given to the functions as a list of `gurobipy.Var`
    N : int
        Number of iterations for the desired trajectory, including the initial
        value

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    """
    x = _add_trapez_constr(m, l, system, N, None)

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
    """Adds a MILP representation of the FO IVP to a `gurobi` model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : :class:`femformal.core.system.FOSystem`
    x0 : array_like, shape ([system order], state dimension, [dofs])
        Initial value.
    N : int
        Number of iterations for the desired trajectory, including the initial
        value

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    """
    x = _add_trapez_constr(m, l, system, N, xhist)
    d0 = x0
    for i in range(system.n):
        m.addConstr(x[label(l, i, 0)] == d0[i])
        if hasattr(system, 'f_nodal'):
            for j in range(N):
                m.addConstr(x[label('f', i, j)] == system.f_nodal(j * system.dt)[i])
        else:
            for j in range(N):
                m.addConstr(x[label('f', i, j)] == 0)
    return x

alpha = 0.5

def _add_trapez_constr(m, l, system, N, xhist=None):
    if xhist is None:
        xhist = []
    M, F, dt = system.M, system.F, system.dt

    # Decision variables
    x = _add_fosys_variables(m, l, M.shape[0], N, len(xhist))

    if hasattr(system, "K_global"):
        deltas = _add_hybrid_K_deltas(m, system.K_els(), x, l, N, system.bigN)
        el_int_forces = [_elements_int_force(
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
                                _int_force(system, x, i, j, l, el_int_forces))
            elif j > 0:
                # d = d + dt * v
                m.addConstr(x[label(l, i, j)] == x[label(l, i, j - 1)] +
                            dt * x[label('d' + l, i, j - 1)])
                # td = d + (1 - alpha) * dt * v
                # m.addConstr(x[label('t' + l, i, j)] == x[label(l, i, j - 1)] +
                #             (1 - alpha) * dt * x[label('d' + l, i, j - 1)])
                # M v = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('d' + l, i, j)] == 0)
                else:
                    m.addConstr(g.quicksum(M[i, k] * x[label('d' + l, k, j)]
                                        for k in range(M.shape[0])) ==
                                x[label('f', i, j)] + F[i] -
                                _int_force(system, x, i, j, l, el_int_forces))
                # (M + alpha * dt * K) d = alpha * dt * F + M td
                # if M[i,i] == 0:
                #     m.addConstr(x[label('d' + l, i, j)] == 0)
                # else:
                #     m.addConstr(g.quicksum(M[i, k] * x[label('d' + l, k, j)]
                #                         for k in range(M.shape[0])) ==
                #                 x[label('f', i, j)] + F[i] -
                #                 int_force(system, x, i, j, l, el_int_forces))
    m.update()

    return x


def add_newmark_constr_x0_set(m, l, system, pset, f, N, beta=0.25, gamma=0.5):
    """Adds the parameterized SO system trajectory to a `gurobi` model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : :class:`femformal.core.system.SOSystem`
    pset : list of array_like, (pd, pv, pf)
        Polytopes of the initial value and nodal foce parameters.
    f : list of callable, (fd, fv, ff)
        Parameterized initial value and nodal force functions. `fd` and `fv`
        must map a point in the domain and the list of parameters to the initial
        value and velocity at that point. `ff` must map a time, the list of
        parameters and a point of the domain to the nodal force at that point
        and time. The parameters are given to the functions as a list of
        `gurobipy.Var`
    N : int
        Number of iterations for the desired trajectory, including the initial
        value
    beta : float, optional
    gamma : float, optional

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    See Also
    --------
    femformal.core.system.newm_integrate

    """
    x = _add_newmark_constr(m, l, system, N, None)

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

def add_newmark_constr_x0(m, l, system, x0, N, xhist=None, beta=0.25, gamma=0.5):
    """Adds a MILP representation of the SO IVP to a `gurobi` model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : :class:`femformal.core.system.SOSystem`
    x0 : array_like, shape ([system order], state dimension, [dofs])
        Initial value.
    N : int
        Number of iterations for the desired trajectory, including the initial
        value
    beta : float, optional
    gamma : float, optional

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    See Also
    --------
    femformal.core.system.newm_integrate

    """
    x = _add_newmark_constr(m, l, system, N, xhist)
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

def _add_newmark_constr(m, l, system, N, xhist=None, beta=0.25, gamma=0.5,
                        use_lu_decomp=False):
    if xhist is None:
        xhist = []
    M, F, dt = system.M, system.F, system.dt

    # Decision variables
    x = _add_sosys_variables(m, l, M.shape[0], N, len(xhist),
                             dpred=True, vpred=True, use_lu_decomp=use_lu_decomp)

    if hasattr(system, "K_global"):
        deltas = _add_hybrid_K_deltas(m, system.K_els(), x, l, N, system.bigN)
        el_int_forces = [_elements_int_force(
            m, l, system.K_els(), x, deltas, time, system.bigN)
            for time in range(N)]
    else:
        el_int_forces = None
        K = system.K

    if use_lu_decomp:
        matrix = (M + beta * dt * dt * K).tocsc()
        lu = spla.splu(matrix, permc_spec='NATURAL')
        # lu_pr = csc_matrix(lu.L.shape)
        # lu_pc = csc_matrix(lu.L.shape)
        # lu_pr[lu.perm_r, np.arange(lu.L.shape[0])] = 1
        # lu_pc[np.arange(lu.L.shape[0]), lu.perm_c] = 1
        # lu_l = lu_pr.T * lu.L
        # lu_u = lu.U * lu_pc.T
        lu_l = lu.L
        lu_u = lu.U

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
                                _int_force(system, x, i, j, l, el_int_forces))
            elif j > 0:
                # td = d + dt * v + 0.5 * dt * dt * (1 - 2 * beta) * a
                m.addConstr(x[label('t' + l, i, j)] == x[label(l, i, j - 1)] +
                            dt * x[label('d' + l, i, j - 1)] + 0.5 * dt * dt *
                            (1 - 2 * beta) * x[label('dd' + l, i, j - 1)])

                # tv = v + (1 - gamma) * dt * a
                m.addConstr(x[label('td' + l, i, j)] == x[label('d' + l, i, j - 1)] +
                            (1 - gamma) * dt * x[label('dd' + l, i, j - 1)])

                # (M + beta * dt * dt * K) a = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('dd' + l, i, j)] == 0)
                else:
                    if use_lu_decomp:
                        m.addConstr(g.quicksum(
                            lu_u[i,k] * x[label('dd' + l, k, j)]
                            for k in range(M.shape[0])) ==
                            x[label('int_dd' + l, i, j)])
                        m.addConstr(g.quicksum(
                            lu_l[i, k] * x[label('int_dd' + l, k, j)]
                            for k in range(M.shape[0]))
                            == x[label('f', i, j)] + F[i] -
                            _int_force(system, x, i, j, "t" + l, el_int_forces))
                    else:
                        m.addConstr(g.quicksum(
                            (M[i, k] + beta * dt * dt * K[i, k])
                            * x[label('dd' + l, k, j)] for k in range(M.shape[0]))
                            == x[label('f', i, j)] + F[i] -
                            _int_force(system, x, i, j, "t" + l, el_int_forces))

                # d = td + beta * dt * dt * a
                m.addConstr(x[label(l, i, j)] == x[label('t' + l, i, j)] +
                            beta * dt * dt * x[label('dd' + l, i, j)])

                # v = tv + gamma * dt * a
                m.addConstr(x[label('d' + l, i, j)] == x[label('td' + l, i, j)] +
                            gamma * dt * x[label('dd' + l, i, j)])
    m.update()

    return x


def add_central_diff_constr_x0_set(m, l, system, pset, f, N):
    """Adds the parameterized SO system trajectory to a `gurobi` model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : :class:`femformal.core.system.SOSystem`
    pset : list of array_like, (pd, pv, pf)
        Polytopes of the initial value and nodal foce parameters.
    f : list of callable, (fd, fv, ff)
        Parameterized initial value and nodal force functions. `fd` and `fv`
        must map a point in the domain and the list of parameters to the initial
        value and velocity at that point. `ff` must map a time, the list of
        parameters and a point of the domain to the nodal force at that point
        and time. The parameters are given to the functions as a list of
        `gurobipy.Var`
    N : int
        Number of iterations for the desired trajectory, including the initial
        value

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    """
    x = _add_central_diff_constr(m, l, system, N, None)

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

def add_central_diff_constr_x0(m, l, system, x0, N, xhist=None):
    """Adds a MILP representation of the SO IVP to a `gurobi` model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
    l : str
        Prefix for the labels used as state variables and auxiliary variables
    system : :class:`femformal.core.system.SOSystem`
    x0 : array_like, shape ([system order], state dimension, [dofs])
        Initial value.
    N : int
        Number of iterations for the desired trajectory, including the initial
        value

    Returns
    -------
    dict
        Dictionary mapping variable labels to :class:`gurobipy.Var` for all
        variables added to the model

    """
    x = _add_central_diff_constr(m, l, system, N, xhist)
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

def _add_central_diff_constr(m, l, system, N, xhist=None):
    if xhist is None:
        xhist = []
    M, F, dt = system.M, system.F, system.dt

    # Decision variables
    x = _add_sosys_variables(m, l, M.shape[0], N, len(xhist))

    if hasattr(system, "K_global"):
        deltas = _add_hybrid_K_deltas(m, system.K_els(), x, l, N, system.bigN)
        el_int_forces = [_elements_int_force(
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
                                _int_force(system, x, i, j, l, el_int_forces))
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
                                _int_force(system, x, i, j, l, el_int_forces))
                # v = tv + .5 * dt * a
                m.addConstr(x[label('d' + l, i, j)] == x[label('td' + l, i, j)] +
                            .5 * dt * x[label('dd' + l, i, j)])
    m.update()

    return x


def _elements_int_force(m, l, kels, x, deltas, time, bigN):
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

def _int_force(system, x, row, time, l, el_int_forces=None):
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


def _add_sosys_variables(m, l, nvars, horlen, histlen, dpred=False, vpred=True,
                         use_lu_decomp=True):
    logger.debug("Adding decision variables")
    x = {}
    bds = [-g.GRB.INFINITY, g.GRB.INFINITY]
    for i in range(nvars):
        for j in range(-histlen, horlen):
            labelf = label('f', i, j)
            x[labelf] = m.addVar(
                obj=0, lb=bds[0], ub=bds[1], name=labelf)
            labelx = label(l, i, j)
            x[labelx] = m.addVar(
                obj=0, lb=bds[0], ub=bds[1], name=labelx)
            labelv = label('d' + l, i, j)
            x[labelv] = m.addVar(
                obj=0, lb=bds[0], ub=bds[1], name=labelv)
            if vpred:
                labeltv = label('td' + l, i, j)
                x[labeltv] = m.addVar(
                    obj=0, lb=bds[0], ub=bds[1], name=labeltv)
            if dpred:
                labeltd = label('t' + l, i, j)
                x[labeltd] = m.addVar(
                    obj=0, lb=bds[0], ub=bds[1], name=labeltd)
            labela = label('dd' + l, i, j)
            x[labela] = m.addVar(
                obj=0, lb=bds[0], ub=bds[1], name=labela)
            if use_lu_decomp:
                labela = label('int_dd' + l, i, j)
                x[labela] = m.addVar(
                    obj=0, lb=bds[0], ub=bds[1], name=labela)

    m.update()
    return x

def _add_fosys_variables(m, l, nvars, horlen, histlen):
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
            # labeltv = label('t' + l, i, j)
            # x[labeltv] = m.addVar(
            #     obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labeltv)
    m.update()
    return x

def _add_hybrid_K_deltas(m, Ks, x, l, N, bigN):
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
    """Obtains the trajectory of a system from a solved model

    Parameters
    ----------
    m : :class:`gurobipy.Model`
        A solved model with a feasible solution
    l : str
        Prefix of the state variable labels
    T : int
        Number of desired iterations of the trajectory
    system : any `System` class in :mod:`femformal.core.system`

    Returns
    -------
    array, shape (T, system.n)
        The system trajectory obtained as a solution to the model

    """
    return np.array([
        [m.getVarByName(label(l, i, j)).x for i in range(system.n)]
        for j in range(T)])
