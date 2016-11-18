import numpy as np
import femformal.system as s
import femformal.util as u
import femmilp.system_milp as sysmilp

def heatlinfem(N, L, T):
    n = N
    l = L / n

    M = np.diag([5.0] + [6.0 for i in range(n - 3)] + [5.0]) * l / 6
    K = (np.diag([2.0 for i in range(n - 1)]) +
        np.diag([-1.0 for i in range(n - 2)], 1) +
        np.diag([-1.0 for i in range(n - 2)], -1)) / l
    F = np.r_[T[0], [0 for i in range(n - 3)], T[1]] / l
    F.shape = (n - 1, 1)

    A = np.linalg.solve(M, -K)
    b = np.linalg.solve(M, F)
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    tpart = [np.arange(5, 115, 10.0).tolist() for i in range(n-1)]
    xpart = np.linspace(0, L, N + 1)

    return system, xpart, tpart

def diag(n, m, i):
    d = np.hstack([j * np.ones((n - 1) / m, dtype=int) for j in range(m)])
    d = np.hstack([d, (m - 1) * np.ones((n - 1) % m, dtype=int)])

    d = d + i
    if i > 0:
        d[d > m - 1] = m - 1
    elif i < 0:
        d[d < 0] = 0

    return d

def build_cs(N, L, T, dt, d0, cregions, cspec):
    system, xpart, partition = heatlinfem(N, L, T)
    system = s.cont_to_disc(system, dt)

    regions = {label: u.ap_cont_to_disc(pred, xpart)
               for label, pred in cregions.items()}
    dspec = u.subst_spec_labels_disc(cspec, regions)
    spec = sysmilp.stl_parser().parseString(dspec)[0]
    sysmilp.scale_time(spec, dt)

    rh_N = 2

    return CaseStudy(system, xpart, T, dt, d0, regions, spec, rh_N)


class CaseStudy(object):

    def __init__(self, system, xpart, T, dt, d0, regions, spec, rh_N):
        self.system = system
        self.xpart = xpart
        self.T = T
        self.dt = dt
        self.d0 = d0
        self.regions = regions
        self.spec = spec
        self.rh_N = rh_N

