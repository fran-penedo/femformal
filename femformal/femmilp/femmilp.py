import logging

from stlmilp import milp_util as milp, stl_milp_encode as stl_milp

from femformal.core import system_milp_encode as sys_milp


logger = logging.getLogger(__name__)

def _build_and_solve(*args, **kwargs):
    m = stl_milp.build_and_solve(*args, **kwargs)
    if m.status == stl_milp.GRB.status.INFEASIBLE:
        logger.warning("MILP infeasible, logging IIS")
        m.computeIIS()
        m.write("out.ilp")
    return m

def verify_singleton(system, d0, spec, fdt_mult=1, **kwargs):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0(m, "d", system, d0, fdt_mult * hd, None)

    m = _build_and_solve(spec, model_encode_f, 1.0, **kwargs)
    return m.getVarByName("spec").getAttr("x")

def verify_set(system, pset, f, spec, fdt_mult=1, **kwargs):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(m, "d", system, pset, f, fdt_mult * hd)

    m = _build_and_solve(spec, model_encode_f, 1.0, **kwargs)
    return m.getVarByName("spec").getAttr("x")

def synthesize(system, pset, f, spec, fdt_mult=1, return_traj=False, T=None, **kwargs):
    if spec is None:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(
            m, "d", system, pset, f, T + 1)
    else:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(
            m, "d", system, pset, f, fdt_mult * hd)

    m = _build_and_solve(spec, model_encode_f, -1.0, **kwargs)
    if m.status == milp.GRB.status.INFEASIBLE:
        return None, None
    if isinstance(f[-1].ys[0], list):
        inputs = [[y.getAttr('x') for y in yy] for yy in f[-1].ys]
    else:
        inputs = [y.getAttr('x') for y in f[-1].ys]
    try:
        robustness = m.getVarByName("spec").getAttr("x")
    except:
        robustness = None
    if return_traj:
        return (robustness, inputs), \
            sys_milp.get_trajectory_from_model(m, "d", T + 1, system)
    else:
        return robustness, inputs

def simulate_trajectory(system, d0, T, pset=None, f=None, labels=None, **kwargs):
    if labels is None:
        labels = ["d"]

    if pset is None:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0(m, "d", system, d0, T + 1, None)
    else:
        model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(
            m, "d", system, pset, f, T + 1)

    m = _build_and_solve(None, model_encode_f, 1.0, **kwargs)
    if len(labels) == 1:
        return sys_milp.get_trajectory_from_model(m, labels[0], T + 1, system)
    else:
        return [sys_milp.get_trajectory_from_model(m, l, T + 1, system) for
                l in labels]

