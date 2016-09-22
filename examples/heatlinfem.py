import numpy as np
import femformal.system as s

def heatlinfem(N):
    L = N * 1.0
    n = N
    l = L / n

    T = [10.0, 100.0]

    M = np.diag([5.0 for i in range(n - 1)]) * l / 6
    K = (np.diag([2.0 for i in range(n - 1)]) +
        np.diag([-1.0 for i in range(n - 2)], 1) +
        np.diag([-1.0 for i in range(n - 2)], -1)) / l
    F = np.r_[T[0], [0 for i in range(n - 3)], T[1]]
    F.shape = (n - 1, 1)

    A = np.linalg.solve(M, -K)
    b = np.linalg.solve(M, F)
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    partition = [np.arange(5, 105, 10.0).tolist() for i in range(n-1)]

    return system, partition

def diag(n, m, i):
    d = np.hstack([j * np.ones((n - 1) / m, dtype=int) for j in range(m)])
    d = np.hstack([d, (m - 1) * np.ones((n - 1) % m, dtype=int)])

    d = d + i
    if i > 0:
        d[d > m - 1] = m - 1
    elif i < 0:
        d[d < 0] = 0

    return d
