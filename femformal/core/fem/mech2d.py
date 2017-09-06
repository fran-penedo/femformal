import logging
from itertools import product as cartesian_product

import numpy as np
import scipy

from . import mesh as mesh
from . import element as element
from .. import system as sys


logger = logging.getLogger(__name__)


def mech2d(xpart, ypart, rho, C, g, f_nodal, dt):
    mesh = grid_mesh(xpart, ypart)
    nnodes = mesh.nnodes
    nelems = mesh.nelems

    # bigk = np.zeros((nnodes*2, nnodes*2))
    # bigm = np.zeros((nnodes*2, nnodes*2))
    bigk = scipy.sparse.lil_matrix((nnodes*2, nnodes*2))
    bigm = scipy.sparse.lil_matrix((nnodes*2, nnodes*2))
    bigf = np.zeros(nnodes*2)

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
            strain_disp = build_elem.strain_displacement(a, b, x, y)
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
            mass += (shape.T.dot(rho * np.identity(2)).dot(shape)
                      * weights[i] * weights[j] / jac_det)

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

