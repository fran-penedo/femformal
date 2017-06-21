import logging

import stlmilp.milp_util as milp
import stlmilp.stl_milp_encode as stl_milp

import femformal.core.system_milp_encode as sys_milp


logger = logging.getLogger('FEMFORMAL')

def _build_and_solve(spec, model_encode_f, spec_obj):
    # print spec
    if spec is not None:
        hd = max(0, spec.horizon())
    else:
        hd = 0

    m = milp.create_milp("rhc_system")
    logger.debug("Adding affine system constraints")
    model_encode_f(m, hd)
    # sys_milp.add_sys_constr_x0(m, "d", system, d0, hd, None)
    if spec is not None:
        logger.debug("Adding STL constraints")
        fvar, vbds = stl_milp.add_stl_constr(m, "spec", spec)
        fvar.setAttr("obj", spec_obj)
    # m.params.outputflag = 0
    # m.params.numericfocus = 3
    m.update()
    m.write("out.lp")
    logger.debug(
        "Optimizing MILP with {} variables ({} binary) and {} constraints".format(
            m.numvars, m.numbinvars, m.numconstrs))
    m.optimize()
    logger.debug("Finished optimizing")
    f = open("out_vars.txt", "w")
    for v in m.getVars():
        print >> f, v
    f.close()

    if m.status != milp.GRB.status.OPTIMAL:
        logger.warning("MILP returned status: {}".format(m.status))
    return m


def verify_singleton(system, d0, spec, fdt_mult=1):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0(m, "d", system, d0, fdt_mult * hd, None)

    m = _build_and_solve(spec, model_encode_f, 1.0)
    return m.getVarByName("spec").getAttr("x")

def verify_set(system, pset, f, spec, fdt_mult=1):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(m, "d", system, pset, f, fdt_mult * hd)

    m = _build_and_solve(spec, model_encode_f, 1.0)
    return m.getVarByName("spec").getAttr("x")

def synthesize(system, pset, f, spec, fdt_mult=1):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0_set(m, "d", system, pset, f, fdt_mult * hd)

    m = _build_and_solve(spec, model_encode_f, -1.0)
    return m.getVarByName("spec").getAttr("x"), [y.getAttr('x') for y in f[-1].ys]

def simulate_trajectory(system, d0, T):
    model_encode_f = lambda m, hd: sys_milp.add_sys_constr_x0(m, "d", system, d0, T + 1, None)

    m = _build_and_solve(None, model_encode_f, 1.0)
    return sys_milp.get_trajectory_from_model(m, "d", T + 1, system)
