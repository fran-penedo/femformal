import stlmilp.milp_util as milp
import stlmilp.stl_milp_encode as stl_milp
import core.system_milp_encode as sys_milp
import core.system as sys

import logging
logger = logging.getLogger('FEMMILP')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s %(module)s:%(lineno)d:%(funcName)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logger.setLevel(logging.DEBUG)

def verify_singleton(system, d0, spec):
    # print spec
    hd = max(0, spec.horizon())

    m = milp.create_milp("rhc_system")
    logger.debug("Adding affine system constraints")
    sys_milp.add_sys_constr_x0(m, "d", system, d0, hd, None)
    logger.debug("Adding STL constraints")
    fvar, vbds = stl_milp.add_stl_constr(m, "spec", spec)
    # m.addConstr(fvar >= 0)
    # m.params.outputflag = 0
    m.update()
    m.write("out.lp")
    logger.debug(
        "Optimizing MILP with {} variables ({} binary) and {} constraints".format(
            m.numvars, m.numbinvars, m.numconstrs))
    m.optimize()
    logger.debug("Finished optimizing")
    # for v in m.getVars():
    #     print v

    if m.status != milp.GRB.status.OPTIMAL:
        logger.warning("MILP returned status: {}".format(m.status))
        return False
    else:
        return fvar.getAttr("x")

def verify_set(system, pset, f, spec):
    # print spec
    hd = max(0, spec.horizon())

    m = milp.create_milp("rhc_system")
    logger.debug("Adding affine system constraints")
    sys_milp.add_sys_constr_x0_set(m, "d", system, pset, f, hd)
    logger.debug("Adding STL constraints")
    fvar, vbds = stl_milp.add_stl_constr(m, "spec", spec)
    fvar.setAttr("obj", 1.0)
    # m.params.outputflag = 0
    # m.params.numericfocus = 3
    m.update()
    # m.write("out.lp")
    logger.debug(
        "Optimizing MILP with {} variables ({} binary) and {} constraints".format(
            m.numvars, m.numbinvars, m.numconstrs))
    m.optimize()
    logger.debug("Finished optimizing")
    # for v in m.getVars():
    #     print v

    if m.status != milp.GRB.status.OPTIMAL:
        logger.warning("MILP returned status: {}".format(m.status))
        return False
    else:
        return fvar.getAttr("x")


