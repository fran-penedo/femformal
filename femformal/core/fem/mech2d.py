import logging
from itertools import product as cartesian_product

import numpy as np
import scipy

from . import mesh as mesh
from . import element as element
from .. import system as sys


logger = logging.getLogger(__name__)


def mech2d(xpart, ypart, rho, C, g, f_nodal, dt, traction=None):
    mesh = grid_mesh(xpart, ypart)
    nnodes = mesh.nnodes
    nelems = mesh.nelems

    # bigk = np.zeros((nnodes*2, nnodes*2))
    # bigm = np.zeros((nnodes*2, nnodes*2))
    bigk = scipy.sparse.lil_matrix((nnodes*2, nnodes*2))
    bigm = scipy.sparse.lil_matrix((nnodes*2, nnodes*2))
    if traction is None:
        bigf = np.zeros(nnodes*2)
    else:
        bigf = traction_nodal_force(traction, mesh)

    for e in range(nelems):
        elem_nodes = mesh.elem_nodes(e)
        x, y = zip(*mesh.nodes_coords[elem_nodes])
        kelem = elem_stiffness(x, y, C, mesh.build_elem)
        assemble_into_big_matrix(bigk, kelem, elem_nodes)
        melem = elem_mass(x, y, rho, mesh.build_elem)
        assemble_into_big_matrix(bigm, melem, elem_nodes)

    bigm = lumped(bigm)

    for node in range(nnodes):
        gnode = g(*mesh.nodes_coords[node])
        for i in range(2):
            if gnode[i] is not None:
                bigm[2*node + i] = 0
                bigk[2*node + i] = 0
                bigk[:, 2*node + i] = 0
                bigk[2*node + i,2*node + i] = 1
                bigf[2*node + i] = gnode[i]

    if f_nodal is not None:
        bigf += f_nodal

    _remove_close_zeros(bigk)
    _remove_close_zeros(bigm)
    bigk = bigk.tocsc()
    bigm = bigm.tocsc()
    sosys = sys.SOSystem(bigm, bigk, bigf, dt=dt, mesh=mesh, build_elem=element.BLQuadQ4)

    return sosys


def _remove_close_zeros(matrix):
    indices = (np.isclose(matrix.A, 0)  & (abs(matrix.A) > 0)).nonzero()
    for i in zip(*indices):
        matrix[i] = 0

def lumped(m):
    return scipy.sparse.diags(np.ravel(m.sum(axis=1)), format='lil')

def grid_mesh(xs, ys):
    return mesh.GridQ4([xs, ys], element.BLQuadQ4)

def assemble_into_big_matrix(matrix, elem_matrix, elem_nodes):
    eqs_grouped = [(2 * x, 2 * x + 1) for x in elem_nodes]
    eqs = [el for p in eqs_grouped for el in p]
    matrix[np.ix_(eqs, eqs)] += elem_matrix

def elem_stiffness(x, y, c, build_elem):
    weights = [1, 1]
    sample_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    stiff = np.zeros((8, 8))

    for i in range(len(weights)):
        for j in range(len(weights)):
            a, b = sample_pts[i], sample_pts[j]
            _, jac_det = build_elem.jacobian(a, b, x, y)
            strain_disp = build_elem.strain_displacement(a, b, x, y) * jac_det
            stiff += (strain_disp.T.dot(c).dot(strain_disp)
                      * weights[i] * weights[j] / jac_det)

    return stiff

def elem_mass(x, y, rho, build_elem):
    weights = [1, 1]
    sample_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    mass = np.zeros((8, 8))

    for i in range(len(weights)):
        for j in range(len(weights)):
            a, b = sample_pts[i], sample_pts[j]
            _, jac_det = build_elem.jacobian(a, b, x, y)
            shape = shape_interp(a, b, build_elem)
            mass += (shape.T.dot(shape)
                      * rho * weights[i] * weights[j] * jac_det)

    return mass

def shape_interp(a, b, build_elem):
    sh = build_elem.shapes(a, b)
    ret = np.zeros((2, 8))
    ret[0, 0:8:2] = sh
    ret[1, 1:8:2] = sh
    return ret

def state(u0, du0, node_coords, g):
    d0 = []
    v0 = []
    for node in node_coords:
        gnode = g(*node)
        u0node = u0(*node)
        du0node = du0(*node)
        for i in range(2):
            if gnode[i] is None:
                d0.append(u0node[i])
                v0.append(du0node[i])
            else:
                d0.append(gnode[i])
                v0.append(0.0)

    return d0, v0

def traction_nodal_force(traction, mesh_):
    f = np.zeros(mesh_.nnodes * 2).tolist()
    for e in mesh_.find_border_elems():
        e_traction = element_traction_nodal_force(traction, e, mesh_)
        ns = mesh_.elem_nodes(e, dim=1)
        for i in [0, 2]:
            f[2*ns[i]] = f[2*ns[i]] + e_traction[i / 2][0]
            f[2*ns[i] + 1] = f[2*ns[i] + 1] + e_traction[i / 2][1]

    return np.array(f)

def element_traction_nodal_force(traction, e, mesh_):
    weights = [1, 1]
    sample_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    coords = mesh_.elem_coords(e, dim=1)
    f = np.zeros((2,2))
    ns = mesh_.elem_nodes(e, dim=1)
    shapes = mesh_.build_elem.shapes
    sample_coords = [shapes(sample_pts[i], sample_pts[i]).dot(coords)
                     for i in range(len(sample_pts))]
    # If vertical
    if ns[0] == ns[1]:
        Ns = [shapes(1, p)[1:3] for p in sample_pts]
        l = np.linalg.norm(coords[2] - coords[1])
    else:
        Ns = [shapes(p, -1)[:2] for p in sample_pts]
        l = np.linalg.norm(coords[1] - coords[0])

    f = [np.sum([weights[i] * traction(*sample_coords[i]) * Ns[i][j] * l / 2
                    for i in range(2)], axis=0)
                for j in range(2)]

    return f

def parabolic_traction(L, c):
    I = 2.0 * (c ** 3) / 3.0
    def traction(x, y, *kargs):
        P, = kargs
        if np.isclose(x, 0):
            return np.array([P * L * y / I, - P * (c * c - y * y) / (2 * I)])
        elif np.isclose(x, L):
            return np.array([0.0, P * (c * c - y * y) / (2 * I)])
        else:
            return np.array([0.0, 0.0])

    return traction


class TimeVaryingTractionForce:
    def __init__(self, parameter, traction_templ, mesh_):
        self.parameter = parameter
        self.traction_templ = traction_templ
        self.mesh_ = mesh_
        self.memoize = {}

    @property
    def ys(self):
        return self._parameter.ys

    def traction_force(self, t):
        if t in self.memoize:
            traction_force = self.memoize[t]
        else:
            traction = lambda x, y: self.traction_templ(x, y, self.parameter(t))
            traction_force = traction_nodal_force(traction, self.mesh_)
            self.memoize[t] = traction_force
        return traction_force

    def __call__(self, t, ys, node):
        self._parameter.ys = ys
        return self.traction_force(t)[node]

