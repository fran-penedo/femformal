import numpy as np
import femformal.system as s

L = 10.0
n = 10
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
v = np.r_[np.arange(1, n-2), n-1].tolist()
regions = {'A': v}
spec = "F (! (state = A))"
init_states = [v]

depth = 3
