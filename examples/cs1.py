import examples.heatlinfem as fem
import femformal.util as u

N = 10
L = 10.0
T = [10.0, 100.0]

system, xpart, partition = fem.heatlinfem(N, L, T)

v = fem.diag(N, 9, 1)
apc = u.APCont([0, L], -1, lambda x: 10*x + 5 if x < 9 else 10*x - 5)
regions = {'A': u.ap_cont_to_disc(apc, xpart)}
spec = "F (! (A))"
init_states = [v]

depth = 3
