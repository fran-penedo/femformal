import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import femmilp.system_milp as sysmilp
import femformal.system as s

N = 10
L = 10.0
T = [10.0, 100.0]

system, xpart, partition = fem.heatlinfem(N, L, T)
dt = .1
system = s.cont_to_disc(system, dt)


v = fem.diag(N, 9, -1)
apc1 = logic.APCont([0, L/2], 1, lambda x: 85)
apc2 = logic.APCont([L/2, L], 1, lambda x: 125)
# apc3 = u.APCont([0, 10], 1, lambda x: 25)
regions = {'A': logic.ap_cont_to_disc(apc1, xpart),
           'B': logic.ap_cont_to_disc(apc2, xpart)}
           # 'C': u.ap_cont_to_disc(apc3, xpart)}

spec_cont ="((G_[0, 10] (A)) & (G_[0, 10] (B)))"
spec_disc = logic.subst_spec_labels_disc(spec_cont, regions)
spec = sysmilp.stl_parser().parseString(spec_disc)[0]
sysmilp.scale_time(spec, dt)

rh_N = 2
d0 = [20.0 for i in range(N + 1)]
d0[0] = T[0]
d0[-1] = T[1]
