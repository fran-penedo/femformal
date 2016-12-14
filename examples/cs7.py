import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np


Ns = [10, 20, 30, 40, 50]
Nlen = len(Ns)
L = 10.0
T = [10.0, 100.0]
dt = .1
d0s = [[T[0]] + [20.0 for i in range(N - 1)] + [T[1]] for N in Ns]

apc1 = logic.APCont([8, 9], -1, lambda x: 32.0 + (60.0 - 32.0) * (x - 8.0))
apc2 = logic.APCont([L/2, L - 1], 1, lambda x: 125)
cregions = {'A': apc1, 'B': apc2}

cspec = "(G_[1, 10] (A))"
# t \in [1,10], T = [10, 100], x \in [1, 9], N = [10, 20, 30, 40, 50], L = 10
eps = [5.10, 1.36, 0.63, 0.35, 0.20]

cstrues = [fem.build_cs(
    1000, L, T, 0.01, [T[0]] + [20.0 for i in range(1000 - 1)] + [T[1]],
    cregions, cspec, discretize_system=False) for N in Ns]

cslist = [fem.build_cs(N, L, T, dt, d0, cregions, cspec, eps=e)
          for N, d0, e in zip(Ns, d0s, eps)]

print "loaded cs"


# No eps
# times: [0.6001558303833008, 2.040926933288574, 4.281876087188721, 7.599426031112671, 11.859421968460083]
# results: [1.6617251024716777, -1.5667232646130744, -2.2885423308672443, -2.683509319906193, -2.7437587278861315]
# true results: [-2.7465919160148786, -2.7465919160148786, -2.7465919160148786, -2.7465919160148786, -2.7465919160148786]


# With eps
# times: [0.596095085144043, 2.010236978530884, 4.18806004524231, 7.5973639488220215, 11.549888849258423]
# results: [-3.438274897527379, -2.926723264612633, -2.91854233086724, -3.0335093199061025, -2.9437587278860633]
# true results: [-2.7465919160148786, -2.7465919160148786, -2.7465919160148786, -2.7465919160148786, -2.7465919160148786]
