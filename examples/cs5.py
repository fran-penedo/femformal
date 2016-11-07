import examples.heatlinfem as fem
import femformal.util as u

N = 1000
L = 1000.0
T = [10.0, 100.0]

system, xpart, partition = fem.heatlinfem(N, L, T)

v = fem.diag(N, 9, -1)
apc1 = u.APCont([500, 500], 1, lambda x: 85)
apc2 = u.APCont([500, 500], 1, lambda x: 35)
apc3 = u.APCont([0, 1000], 1, lambda x: 25)
regions = {'A': u.ap_cont_to_disc(apc1, xpart),
           'B': u.ap_cont_to_disc(apc2, xpart),
           'C': u.ap_cont_to_disc(apc3, xpart)}
spec = "F G (A)"
init_states = [v]

depth = 1
