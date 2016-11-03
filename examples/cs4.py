import examples.heatlinfem as fem
import femformal.util as u
import femmilp.system_milp as sysmilp

N = 10
L = 10.0
T = [10.0, 100.0]

system, xpart, partition = fem.heatlinfem(N, L, T)

v = fem.diag(N, 9, -1)
apc1 = u.APCont([0, 5], 1, lambda x: 85)
apc2 = u.APCont([5, 10], 1, lambda x: 125)
apc3 = u.APCont([0, 10], 1, lambda x: 25)
regions = {'A': u.ap_cont_to_disc(apc1, xpart),
           'B': u.ap_cont_to_disc(apc2, xpart),
           'C': u.ap_cont_to_disc(apc3, xpart)}

spec_cont = "G_[0, 10] (A) & G_[0, 10] (B)"
spec_disc = u.subst_spec_labels_disc(spec_cont, regions)
spec = sysmilp.stl_parser().parseString(spec_disc)[0]

rh_N = 100
d0 = [20.0 for i in range(N - 1)]

