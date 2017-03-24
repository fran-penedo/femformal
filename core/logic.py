import itertools as it
from bisect import bisect_left, bisect_right
from .util import state_label

class APCont(object):
    def __init__(self, A, r, p, dp = None, uderivs = 0):
        # A : [x_min, x_max] (np.array)
        self.A = A
        # r == 1: f < p, r == -1: f > p
        if r == "<":
            self.r = 1
        elif r == ">":
            self.r = -1
        else:
            self.r = r
        self.p = p
        self.uderivs = uderivs
        if dp:
            self.dp = dp
        else:
            self.dp = lambda x: 0

class APDisc(object):
    def __init__(self, r, m, isnode, uderivs = 0):
        # r == 1: f < p, r == -1: f > p
        self.r = r
        self.isnode = isnode
        # m : i -> (p((x_i + x_{i+1})/2) if not isnode else i -> p(x_i),
        # dp(.....))
        self.m = m
        self.uderivs = uderivs

    def __str__(self):
        return "({})".format(" & ".join(
            ["({isnode} {uderivs} {index} {op} {p} {dp})".format(
                isnode="d" if self.isnode else "y", uderivs=self.uderivs,
                index=i, op="<" if self.r == 1 else ">",
                p=p, dp=dp) for (i, (p, dp)) in self.m.items()]))





# xpart : [x_i] (list)
# FIXME TS based approach probably has wrong index assumption
def ap_cont_to_disc(apcont, xpart):
    r = apcont.r
    N1 = len(xpart)
    if apcont.A[0] == apcont.A[1]:
        if apcont.uderivs > 0:
            raise Exception("Derivatives at nodes are not well defined")
        i = min(max(bisect_left(xpart, apcont.A[0]), 0), N1 - 1)
        m = {i - 1: (apcont.p(xpart[i]), apcont.dp(xpart[i]))}
        isnode = True
    else:
        if apcont.uderivs > 1:
            raise Exception(
                ("Second and higher order derivatives are 0 for linear "
                "interpolation: uderivs = {}").format(apcont.uderivs))

        i_min = max(bisect_left(xpart, apcont.A[0]), 0)
        i_max = min(bisect_left(xpart, apcont.A[1]), N1 - 1)
        m = {i : (apcont.p((xpart[i] + xpart[i+1]) / 2.0),
                  apcont.dp((xpart[i] + xpart[i+1]) / 2.0))
             for i in range(i_min, i_max)}
        isnode = False
    return APDisc(r, m, isnode, apcont.uderivs)

def project_apdisc(apdisc, indices, tpart):
    state_indices = []
    for i in indices:
        if i in apdisc.m:
            if apdisc.r == 1:
                bound_index = bisect_right(tpart, apdisc.m[i]) - 1
                state_indices.append(list(range(bound_index)))
            if apdisc.r == -1:
                bound_index = bisect_left(tpart, apdisc.m[i])
                state_indices.append(list(range(bound_index, len(tpart) - 1)))

    return list(it.product(*state_indices))

def subst_spec_labels_disc(spec, regions):
    res = spec
    for k, v in regions.items():
        replaced = str(v)
        res = res.replace(k, replaced)
    return res


def subst_spec_labels(spec, regions):
    res = spec
    for k, v in regions.items():
        if any(isinstance(el, list) or isinstance(el, tuple) for el in v):
            replaced = "(" + " | ".join(["(state = {})".format(state_label(s))
                                          for s in v]) + ")"
        else:
            replaced = "(state = {})".format(state_label(v))
        res = res.replace(k, replaced)
    return res

def project_apdict(apdict, indices, tpart):
    ret = {}
    for key, value in apdict.items():
        ret[key] = project_apdisc(value, indices, tpart)
    return ret

