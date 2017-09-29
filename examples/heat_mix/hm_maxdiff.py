import numpy as np

from femformal.core.fem.heatlinfem import heatlinfem_mix
from femformal.core import casestudy, system as sys
from examples.heat_mix.hm_model import *

n_its = 2

u0 = lambda x: 300.0

mult_t = 10
N_t = mult_t * N
xpart_t = np.linspace(0, L, N_t + 1)
f_nodal_t = np.zeros(N_t + 1)
dt_t = 0.001

fosys = sys.ControlFOSystem.from_fosys(heatlinfem_mix(xpart, rho, E, g, f_nodal, dt), None)
fosys_t = sys.ControlFOSystem.from_fosys(heatlinfem_mix(xpart_t, rho, E, g, f_nodal_t, dt_t), None)

def f_unif_sample(bounds, g, xpart_x, xpart_y=None):
    xs = bounds['xs']
    ybounds = bounds['ybounds']
    ys = [np.random.rand() * (ybounds[1] - ybounds[0]) + ybounds[0] for x in xs]
    pwlf = sys.PWLFunction(xs, ys)
    def f_x(t):
        f = np.zeros(len(xpart_x))
        f[-1] = pwlf(t)
        return f
    if xpart_y is not None:
        def f_y(t):
            f = np.zeros(len(xpart_y))
            f[-1] = pwlf(t)
            return f
        return f_x, f_y
    else:
        return f_x


t0 = 0.0
T = 5.0
input_dt = 0.5
tlims = [int(round(t0 / dt)) * dt, int(round(T / dt)) * dt]
xlims = [0.0, L]
fbounds = {'xs': np.linspace(0, T, round(T / input_dt) + 1),
           'ybounds': [0.0, 1e5]}

mds = casestudy.max_diff(fosys, g, tlims, xlims, fosys_t,
                         (u0, fbounds), (casestudy.id_sample_1dof, f_unif_sample),
                         n=n_its, pw=True)
mdxs = casestudy.max_xdiff(fosys, g, tlims,
                     (u0, fbounds), [casestudy.id_sample_1dof, f_unif_sample],
                     n=n_its)
mdtxs = casestudy.max_tdiff(fosys, g, tlims,
                      (u0, fbounds), [casestudy.id_sample_1dof, f_unif_sample],
                      n=n_its)

print "eps = {}".format(mds.__repr__())
print "eta = {}".format(mdxs.__repr__())
print "nu = {}".format(mdtxs.__repr__())
