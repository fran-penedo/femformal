import numpy as np

from femformal.core import system as sys, casestudy as casestudy
from femformal.core.fem.mechlinfem import mechlinfem
from examples.mech_mix.mm_model import *


n_its = 100

u0 = lambda x: 0.0
v0 = lambda x: 0.0

sosys = sys.ControlSOSystem.from_sosys(
    mechlinfem(xpart, rho, E, g, f_nodal, dt), None)
sosys_t = sys.ControlSOSystem.from_sosys(
    mechlinfem(xpart_t, rho, E, g, f_nodal_t, dt_t), None)


__SAMPLE_PREDEF = True
def f_unif_sample(bounds, g, xpart_x, xpart_y=None):
    global __SAMPLE_PREDEF
    xs = bounds['xs']
    ybounds = bounds['ybounds']
    if __SAMPLE_PREDEF:
        ys = [ybounds[1] for x in xs]
        __SAMPLE_PREDEF = False
    else:
        ys = [np.random.rand() * (ybounds[1] - ybounds[0]) + ybounds[0] for x in xs]
    ys[0] = 0.0
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
input_dt = .1
tlims = [int(round(t0 / dt)) * dt, int(round(T / dt)) * dt]
xlims = [0.0, L]
fbounds = {'xs': np.linspace(0, T, round(T / input_dt) + 1),
           'ybounds': [-5e3, 5e3]}

__SAMPLE_PREDEF = True
mds = casestudy.max_diff(sosys, g, tlims, xlims, sosys_t,
                         ([u0, v0], fbounds), (casestudy.id_sample, f_unif_sample),
                         n=n_its, pw=True)
__SAMPLE_PREDEF = True
mds_xderiv = casestudy.max_diff(sosys, g, tlims, xlims, sosys_t,
                                ([u0, v0], fbounds), [casestudy.id_sample, f_unif_sample],
                                n=n_its, pw=True, xderiv=True, log=False)
__SAMPLE_PREDEF = True
mdxs, mdtxs = casestudy.max_der_diff(sosys, g, tlims,
                     ([u0, v0], fbounds), [casestudy.id_sample, f_unif_sample],
                     n=n_its)
__SAMPLE_PREDEF = True
mdxs_xderiv, mdtxs_xderiv = casestudy.max_der_diff(sosys, g, tlims,
                      ([u0, v0], fbounds), [casestudy.id_sample, f_unif_sample],
                      n=n_its, xderiv=True)

print "eps = {}".format(mds.__repr__())
print "eps_xderiv = {}".format(mds_xderiv.__repr__())
print "eta = {}".format(mdxs.__repr__())
print "nu = {}".format(mdtxs.__repr__())
print "nu_xderiv = {}".format(mdtxs_xderiv.__repr__())
