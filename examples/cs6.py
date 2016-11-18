import examples.heatlinfem as fem
import femformal.util as u


Ns = [10, 20, 30, 40, 50]
Nlen = len(Ns)
Ls = [10.0 for i in range(Nlen)]
Ts = [[10.0, 100.0] for i in range(Nlen)]
dts = [.1 for i in range(Nlen)]
d0s = [[20.0 for i in range(N - 1)] for N in Ns]

L = Ls[0]
apc1 = u.APCont([0, L/2], 1, lambda x: 85)
apc2 = u.APCont([L/2, L], 1, lambda x: 125)
cregionss = [{'A': apc1,
        'B': apc2} for i in range(Nlen)]

cspecs = ["G_[0, 10] (A) G_[0, 10] (B)" for i in range(Nlen)]

cslist = [fem.build_cs(N, L, T, dt, d0, cregions, cspec)
          for N, L, T, dt, d0, cregions, cspec
          in zip(Ns, Ls, Ts, dts, d0s, cregionss, cspecs)]
