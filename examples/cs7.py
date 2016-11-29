import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np


Ns = [50, 50, 50, 50, 50]
Nlen = len(Ns)
Ls = [10.0 for i in range(Nlen)]
Ts = [[10.0, 100.0] for i in range(Nlen)]
dts = [.1 for i in range(Nlen)]
ab = [[np.random.random() * 70 + 20 for i in range(2)] for N in Ns]
d0s = [[a + i * (b - a) / N for i in range(N + 1)] for N, (a,b) in zip(Ns, ab)]

L0 = Ls[0]
apc1 = logic.APCont([1, L0 - 1], -1, lambda x: 10 + 9 * x)
apc2 = logic.APCont([L0/2, L0 - 1], 1, lambda x: 125)
cregionss = [{'A': apc1,
        'B': apc2} for i in range(Nlen)]

cspecs = ["(G_[1, 10] (A))" for i in range(Nlen)]

cslist = [fem.build_cs(N, L, T, dt, d0, cregions, cspec)
          for N, L, T, dt, d0, cregions, cspec
          in zip(Ns, Ls, Ts, dts, d0s, cregionss, cspecs)]

cstrues = [fem.build_cs(1000, L, T, 0.01,
                        [a + (i+1) * (b - a) / 1000 for i in range(1000 - 1)],
                        cregions, cspec, discretize_system=False)
          for L, T, cregions, cspec, (a,b)
          in zip(Ls, Ts, cregionss, cspecs, ab)]

print "loaded cs"
