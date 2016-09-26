import examples.heatlinfem as fem

N = 10

system, partition = fem.heatlinfem(N)

v = fem.diag(N, 9, 1)
regions = {'A': v}
spec = "F (! (state = A))"
init_states = [v]

depth = 3
