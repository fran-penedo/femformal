import stlmilp.milp_util as milp
import stlmilp.stl as stl
from collections import deque
from pyparsing import Word, Literal, MatchFirst, nums

def rh_system_sat(system, d0, N, spec):
    hd = max(0, spec.horizon())
    H = hd
    dcur = d0
    # dhist = deque([xcur])

    for j in range(N - 1):
        m = milp.create_milp("rhc_system")
        d = milp.add_affsys_constr(m, "d", system.A, system.b, dcur, H, None)
        fvar, vbds = milp.add_stl_constr(m, "spec", spec)
        m.addConstr(fvar >= 0)
        m.params.outputflag = 0
        m.update()
        # if j == 0:
        #     m.write("out.lp")
        m.optimize()
        # if j == 0:
        #     for v in m.getVars():
        #         print v

        if m.status != milp.GRB.status.OPTIMAL:
            return False
        else:
            d_opt = m.getAttr("x", d)
            dcur = [d_opt[milp.label("d", i, 1)] for i in range(system.n)]
            # dhist.append(dcur)
            # if j > hd:
            #     dhist.popleft()

    return True

class SysSignal(stl.Signal):
    def __init__(self, index=0, op=stl.LE, p=0):
        self.index = index
        self.op = op
        self.p = p

        self.labels = [lambda t: milp.label("d", self.index, t)]
        self.f = lambda vs: (vs[0] - self.p) * (-1 if self.op == stl.LE else 1)
        self.bounds = [-1000, 1000] #FIXME

def expr_parser():
    num = stl.num_parser()

    T_LE = Literal("<")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    relation = (T_LE | T_GR).setParseAction(lambda t: stl.LE if t[0] == "<" else stl.GT)
    expr = integer + relation + num
    expr.setParseAction(lambda t: SysSignal(t[0], t[1], t[2]))

    return expr

def stl_parser():
    stl_parser = MatchFirst(stl.stl_parser(expr_parser(), False))
    return stl_parser
