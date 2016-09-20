import examples.heatlinfem as fem

N = 100

system, partition = fem.heatlinfem(N)

v = fem.diag(N, 9, 0)
vu = fem.diag(N, 9, 1)
vd = fem.diag(N, 9, -1)
regions = {'A': v, 'B': vu, 'C': vd}
#FIXME doesn't make sense, needs intermediate states
spec = "G ((state = A) | (state = B) | (state = C))"
init_states = [v]

depth = 3
