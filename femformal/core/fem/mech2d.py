import logging
from itertools import product as cartesian_product

import numpy as np
import scipy

from . import foobar as fem
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
        kelem = elem_stiffness(x, y, C)
        assemble_into_big_matrix(bigk, kelem, elem_nodes)
        melem = elem_mass(x, y, rho)
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
    sosys = sys.SOSystem(bigm, bigk, bigf, mesh.nodes_coords, dt)

    return sosys, mesh


def _remove_close_zeros(matrix):
    indices = (np.isclose(matrix.A, 0)  & (abs(matrix.A) > 0)).nonzero()
    for i in zip(*indices):
        matrix[i] = 0


def lumped(m):
    return scipy.sparse.diags(np.ravel(m.sum(axis=1)), format='lil')



def grid_mesh(xs, ys):
    nodes_coords = np.array(sorted(cartesian_product(xs, ys), key=lambda x: x[1]))
    return fem.GridQ4(nodes_coords, (len(xs), len(ys)))


def assemble_into_big_matrix(matrix, elem_matrix, elem_nodes):
    eqs_grouped = [(2 * x, 2 * x + 1) for x in elem_nodes]
    eqs = [el for p in eqs_grouped for el in p]
    matrix[np.ix_(eqs, eqs)] += elem_matrix

def elem_stiffness(x, y, c):
    weights = [1, 1]
    sample_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    stiff = np.zeros((8, 8))

    dshapes = shape_derivatives
    jcob_inv, jcob_det = jacobian(dshapes, x, y)
    strain_disp = strain_displacement(dshapes, jcob_inv)
    integ = integrand(strain_disp, c, jcob_det)

    for i in range(len(weights)):
        for j in range(len(weights)):
            stiff += integ(sample_pts[i], sample_pts[j]) * weights[i] * weights[j]

    return stiff

def elem_mass(x, y, rho):
    weights = [1, 1]
    sample_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    mass = np.zeros((8, 8))

    _, jcob_det = jacobian(shape_derivatives, x, y)
    integ = integrand(shape_interp, rho * np.identity(2), jcob_det)

    for i in range(len(weights)):
        for j in range(len(weights)):
            mass += integ(sample_pts[i], sample_pts[j]) * weights[i] * weights[j]

    return mass

def shapes(a, b):
    return 0.25 * np.array([(1 - a)*(1 - b), (1 + a) * (1 - b),
                            (1 + a) * (1 + b), (1 - a) * (1 + b)])

def shape_interp(a, b):
    sh = shapes(a, b)
    ret = np.zeros((2, 8))
    ret[0, 0:8:2] = sh
    ret[1, 1:8:2] = sh
    return ret

def shape_derivatives(a, b):
    return 0.25 * np.array([[- (1 - b), (1 - b), (1 + b), - (1 + b)],
                            [- (1 - a), - (1 + a), (1 + a), (1 - a)]])

def jacobian(dshapes, x, y):
    jac = lambda a, b: dshapes(a,b).dot(np.vstack([x,y]).T)
    jac_inv = invert_2x2(jac)
    jac_det = lambda a, b: np.linalg.det(jac(a,b))
    return jac_inv, jac_det

def strain_displacement(dshapes, jcob_inv):
    def B(a, b):
        dshapes_phy = jcob_inv(a,b).dot(dshapes(a,b))
        ret = np.zeros((3,8))
        ret[0, 0:8:2] = dshapes_phy[0]
        ret[1, 1:8:2] = dshapes_phy[1]
        ret[2, 0:8:2] = dshapes_phy[1]
        ret[2, 1:8:2] = dshapes_phy[0]
        return ret

    return B

def integrand(strain_disp, c, jcob_det):
    def integ(a, b):
        B = strain_disp(a, b)
        return B.T.dot(c).dot(B) / jcob_det(a, b)
    return integ

def invert_2x2(jac):
    def jac_inv(a, b):
        j = jac(a, b)
        return np.array([[j[1,1], -j[0, 1]], [-j[1,0], j[0,0]]])

    return jac_inv


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

