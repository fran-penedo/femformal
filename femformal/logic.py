import itertools as it
from bisect import bisect_left, bisect_right
from .util import state_label

class APCont(object):
    def __init__(self, A, r, p, dp = None):
        # A : [x_min, x_max] (np.array)
        self.A = A
        # r == 1: f < p, r == -1: f > p
        self.r = r
        self.p = p
        if dp:
            self.dp = dp
        else:
            self.dp = lambda x: 0

class APDisc(object):
    def __init__(self, r, m, isnode):
        # r == 1: f < p, r == -1: f > p
        self.r = r
        self.isnode = isnode
        # m : i -> (p((x_i + x_{i+1})/2) if not isnode else i -> p(x_i),
        # dp(.....))
        self.m = m


# xpart : [x_i] (list)
# FIXME TS based approach probably has wrong index assumption
def ap_cont_to_disc(apcont, xpart):
    r = apcont.r
    N1 = len(xpart)
    if apcont.A[0] == apcont.A[1]:
        i = min(max(bisect_left(xpart, apcont.A[0]), 0), N1 - 1)
        m = {i - 1: (apcont.p(xpart[i]), apcont.dp(xpart[i]))}
        isnode = True
    else:
        i_min = max(bisect_left(xpart, apcont.A[0]), 0)
        i_max = min(bisect_left(xpart, apcont.A[1]), N1 - 1)
        m = {i : (apcont.p((xpart[i] + xpart[i+1]) / 2.0),
                  apcont.dp((xpart[i] + xpart[i+1]) / 2.0))
             for i in range(i_min, i_max)}
        isnode = False
    return APDisc(r, m, isnode)

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
        replaced = "(" + " & ".join(["({} {} {} {} {})".format(
            "d" if v.isnode else "y", i,
            "<" if v.r == 1 else ">", p, dp) for (i, (p, dp)) in v.m.items()]) + ")"
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

