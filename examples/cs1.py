import numpy as np

L = 10.0
n = 10
l = L / n

T = [10.0, 100.0]

M = np.diag([5 for i in range(n - 1)]) * l / 6
K = (np.diag([2 for i in range(n - 1)]) +
     np.diag([-1 for i in range(n - 2)], 1) +
     np.diag([-1 for i in range(n - 2)], -1)) / l
F = np.r_[T[0], [0 for i in range(n - 3)], T[1]]

A = np.lingalg.solve(M, -K)
b = np.lingalg.solve(M, F)
C = np.empty(shape=(0,0))
system = s.System(A, b, C)
partition = [np.arange(5, 105, 10).tolist() for i in range(n-1)]
regions = {'A': list(range(n - 1))}
spec = "F G state = A"
init_states = [list(range(n - 1))]

depth = 3
