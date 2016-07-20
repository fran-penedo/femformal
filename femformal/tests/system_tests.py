import femformal.system as s
import numpy as np

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


