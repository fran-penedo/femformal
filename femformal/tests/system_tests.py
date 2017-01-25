import femformal.system as s
import femformal.util as u
import examples.heatlinfem as fem
import numpy as np

import logging
logger = logging.getLogger('FEMFORMAL')

def system_test():
    A = np.eye(5,5)*4 + np.eye(5,5,k=-1)*3 + np.eye(5,5,k=1)*2
    b = np.array([1,2,3,4,5])
    b = b[:,None]
    i = [0,1,4]
    Ai = np.array([[4, 2, 0], [3, 4, 0], [0, 0, 4]])
    bi = np.array([1,2,5])
    bi = bi[:, None]
    Ci = np.array([[0, 0], [2, 0], [0, 3]])

    S = s.System(A, b)

    assert S.n == 5
    assert S.m == 0

    Ss = S.subsystem(i)

    np.testing.assert_array_equal(Ss.A, Ai)
    np.testing.assert_array_equal(Ss.b, bi)
    np.testing.assert_array_equal(Ss.C, Ci)


def reach_test():
    A = np.array([[-2, 1], [1, -2]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    dist_bounds = np.empty(shape=(0,0))
    R1 = np.array([[-1, 1], [-1, 1]])
    for i in range(2):
        # logger.debug(i)
        facet = R1.copy()
        facet[i, 0] = facet[i, 1]
        assert s.is_facet_separating(system, facet, 1, i, dist_bounds)
        facet = R1.copy()
        facet[i, 1] = facet[i, 0]
        # logger.debug(facet)
        assert s.is_facet_separating(system, facet, -1, i, dist_bounds)

    facet = np.array([[-1, 1], [-0.1, -0.1]])
    assert not s.is_facet_separating(system, facet, 1, 0, dist_bounds)

def reach_facet_test():
    A = np.array([[-2, 1], [1, -2]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    dist_bounds = np.empty(shape=(0,0))
    R1 = np.array([[-1, -1], [-1, 1]])
    assert s.is_facet_separating(system, R1, -1, 0, dist_bounds)

    R2 = np.array([[-1, -1], [1, 2]])
    assert not s.is_facet_separating(system, R2, 1, 0, dist_bounds)

def region_invariant_test():
    A = np.array([[-2, 1], [1, -2]])
    b = np.zeros((2, 1))
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    dist_bounds = np.empty(shape=(0,0))
    R = np.array([[-1, 1], [-1, 1]])
    assert s.is_region_invariant(system, R, dist_bounds)
    R = np.array([[-1, -0.2], [-1, 1]])
    assert not s.is_region_invariant(system, R, dist_bounds)


def cont_disc_test():
    A = np.array([[-2, 1], [1, -2]])
    b = np.array([[1], [2]])
    system = s.System(A, b)
    x0 = np.array([1,2])
    t_cont = np.linspace(0, 10, 101)
    t_disc = 10

    system_d = s.cont_to_disc(system)
    x_c = s.cont_integrate(system, x0, t_cont)
    x_d = s.disc_integrate(system_d, x0, t_disc)
    np.testing.assert_array_almost_equal(x_c[-1], x_d[-1])

    t_cont = np.linspace(0, 10, 101)
    t_disc = 20

    system_d = s.cont_to_disc(system, .5)
    x_c = s.cont_integrate(system, x0, t_cont)
    x_d = s.disc_integrate(system_d, x0, t_disc)
    np.testing.assert_array_almost_equal(x_c[-1], x_d[-1])

    N = 50
    L = 10.0
    T = [10.0, 100.0]
    t_cont = np.linspace(0, 10, 101)
    t_disc = 1000

    system, xpart, partition = fem.heatlinfem(N, L, T)
    system_d = s.cont_to_disc(system, 0.01)
    x0 = [20.0 for i in range(N - 1)]

    x_c = s.cont_integrate(system, x0, t_cont)
    x_d = s.disc_integrate(system_d, x0, t_disc)
    np.testing.assert_array_almost_equal(x_c[-1], x_d[-1], decimal=1)



