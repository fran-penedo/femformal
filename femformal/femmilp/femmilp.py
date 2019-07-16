import logging

from scipy.optimize import differential_evolution
import numpy as np

from stlmilp import milp_util as milp, stl_milp_encode as stl_milp

from femformal.core import logic, system_milp_encode as sys_milp, system as sys


logger = logging.getLogger(__name__)


def _build_and_solve(*args, **kwargs):
    m = stl_milp.build_and_solve(*args, **kwargs)
    if m.status == stl_milp.GRB.status.INFEASIBLE:
        logger.warning("MILP infeasible, logging IIS")
        m.computeIIS()
        m.write("out.ilp")
    return m


def verify_singleton(
    system, d0, spec, fdt_mult=1, start_robustness_tree=None, **kwargs
):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0(
        m, "d", system, d0, fdt_mult * hd, None
    )

    m = _build_and_solve(spec, model_encode_f, 1.0, start_robustness_tree, **kwargs)
    return m.getVarByName("spec").getAttr("x")


def verify_set(system, pset, f, spec, fdt_mult=1, start_robustness_tree=None, **kwargs):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(
        m, "d", system, pset, f, fdt_mult * hd
    )

    m = _build_and_solve(spec, model_encode_f, 1.0, start_robustness_tree, **kwargs)
    return m.getVarByName("spec").getAttr("x")


def _initial_values(system, pset, ff):
    d0 = []
    for p, f in zip(pset[:-1], ff[:-1]):
        if p.shape != (2, 2) or np.any(p[:, 0] != [1, -1]) or p[1, 1] != -p[0, 1]:
            raise NotImplementedError()
        par = [p[0, 1]]
        if system.xpart is not None:
            d0.append([f(system.xpart[i], par) for i in range(len(system.xpart))])
        else:
            raise NotImplementedError()
    if len(d0) == 1:
        d0 = d0[0]

    bounds = []
    fset = pset[-1]
    nbounds = fset.shape[0] // 2
    for i in range(nbounds):
        if fset[i, i] != 1 or fset[nbounds + i, i] != -1:
            raise NotImplementedError()
        bounds.append([-fset[nbounds + i, -1], fset[i, -1]])

    return d0, bounds, ff[-1]


def find_good_start_vector(system, pset, f, spec, objective):
    logger.info("Attempting to find a good starting vector")
    try:
        d0, bounds, pwlf = _initial_values(system, pset, f)
    except NotImplementedError:
        logger.warning("Initial values not supported for start vector search")
        return None

    if system.xpart is not None:
        N = len(system.xpart) - 1
    else:
        # Won't get to this, FIXME when it does
        N = system.mesh.nnodes - 1

    def f_nodal_control(t):
        f = np.zeros(N + 1)
        f[-1] = pwlf(t, x=pwlf.x)
        return f

    csys = sys.make_control_system(system, f_nodal_control)

    def obj(x, *args):
        pwlf.ys = x
        return objective * logic.csystem_robustness(spec, csys, d0, tree=False)

    res = differential_evolution(obj, bounds, maxiter=50, disp=True)
    pwlf.ys = res.x

    tree = logic.csystem_robustness(spec, csys, d0, tree=True)
    pwlf.ys = None
    return tree


def synthesize(
    system,
    pset,
    f,
    spec,
    fdt_mult=1,
    return_traj=False,
    T=None,
    start_robustness_tree=None,
    **kwargs
):
    if spec is None:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(
            m, "d", system, pset, f, T + 1
        )
    else:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(
            m, "d", system, pset, f, fdt_mult * hd
        )
        if start_robustness_tree is None:
            start_robustness_tree = find_good_start_vector(system, pset, f, spec, -1)

    m = _build_and_solve(spec, model_encode_f, -1.0, start_robustness_tree, **kwargs)
    if m.status == milp.GRB.status.INFEASIBLE:
        return None, None
    if isinstance(f[-1].ys[0], list):
        inputs = [[y.getAttr("x") for y in yy] for yy in f[-1].ys]
    else:
        inputs = [y.getAttr("x") for y in f[-1].ys]
    try:
        robustness = m.getVarByName("spec").getAttr("x")
    except:
        robustness = None
    if return_traj:
        return (
            (robustness, inputs),
            sys_milp.get_trajectory_from_model(m, "d", T + 1, system),
        )
    else:
        return robustness, inputs


def simulate_trajectory(system, d0, T, pset=None, f=None, labels=None, **kwargs):
    if labels is None:
        labels = ["d"]

    if pset is None:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0(
            m, "d", system, d0, T + 1, None
        )
    else:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(
            m, "d", system, pset, f, T + 1
        )

    m = _build_and_solve(None, model_encode_f, 1.0, **kwargs)
    if len(labels) == 1:
        return sys_milp.get_trajectory_from_model(m, labels[0], T + 1, system)
    else:
        return [sys_milp.get_trajectory_from_model(m, l, T + 1, system) for l in labels]
