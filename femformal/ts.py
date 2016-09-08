import networkx as nx
import numpy as np
import itertools as it
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
from femformal.util import state_label, subst_spec_labels
from femformal.system import is_facet_separating

import logging
logger = logging.getLogger('FEMFORMAL')

class TS(nx.DiGraph):

    def __init__(self):
        nx.DiGraph.__init__(self)

    def add_transition(self, x, y, pert_bounds):
        xs, ys = label_state(x), label_state(y)
        xs, ys = long_first(xs, ys)
        xn = self.node[xs]
        R = xn['rect'].copy()

        if xs == ys:
            if is_region_invariant(xn['system'], R, pert_bounds):
                ts.add_edge(state_label(xs), state_label(xs))
        else:
            dim, normal = first_change(xs, ys)

            if normal == -1:
                R[dim][1] = R[dim][0]
            else:
                R[dim][0] = R[dim][1]
            if not is_facet_separating(xn['system'], R, normal, dim, pert_bounds):
                ts.add_edge(state_label(xs), state_label(ys))


    def toNUSMV(self, spec, regions, init):
        out = StringIO()
        print >>out, 'MODULE main'
        print >>out, 'VAR'
        print >>out, 'state : {};'.format(_nusmv_statelist(self.nodes()))
        print >>out, 'ASSIGN'
        print >>out, 'init(state) := {};'.format(
            _nusmv_statelist([self.nodes()[i] for i in init]))
        print >>out, 'next(state) := '
        print >>out, 'case'

        for node in self.nodes():
            succ = self.successors(node)
            if len(succ) > 0:
                print >>out, 'state = {} : {};'.format(
                    node,
                    _nusmv_statelist(succ))

        print >>out, 'TRUE : state;'
        print >>out, 'esac;'
        if spec is not None:
            print >>out, 'LTLSPEC {}'.format(subst_spec_labels(spec, regions))

        s = out.getvalue()
        out.close()
        return s


def _nusmv_statelist(l):
    return '{{{}}}'.format(', '.join(l))


def state_n(ts, state):
    return ts.nodes().index(state_label(state))

def abstract(system, partition, pert_bounds):
    if len(partition) + len(pert_bounds) != system.n + system.m:
        raise Exception("System dimensions do not agree with partition")

    indices = [list(range(len(p) - 1)) for p in partition]
    states = list(it.product(*indices))
    nodes = [(state_label(s),
              {'rect': np.array([[p[i], p[i+1]] for p, i in zip(partition, s)]),
               'state': s,
               'system': system})
             for s in states]
    ts = TS()
    ts.add_nodes_from(nodes)
    for s in states:
        for i in range(system.n):
            if s[i] > 0:
                R = ts.node[state_label(s)]['rect'].copy()
                R[i][1] = R[i][0]
                if not is_facet_separating(system, R, -1, i, pert_bounds):
                    nex = list(s)
                    nex[i] -= 1
                    ts.add_edge(state_label(s), state_label(nex))

            if s[i] < len(partition[i]) - 1:
                R = ts.node[state_label(s)]['rect'].copy()
                R[i][0] = R[i][1]
                if not is_facet_separating(system, R, 1, i, pert_bounds):
                    nex = list(s)
                    nex[i] += 1
                    ts.add_edge(state_label(s), state_label(nex))
    ws = np.array([[p[0], p[len(p) - 1]] for p in partition])
    for i in range(system.n):
        R = ws.copy()
        R[i][1] = R[i][0]
        R[i][0] = -np.infty
        index = [0 for j in range(system.n)]
        index[i] = -1
        ts.add_node(state_label(index), rect=R, state=index, system=system)
        for s in it.product(*[list(range(len(p) - 1))
                              for p in (partition[:i] + [[0, 1]] +
                                        partition[i+1:])]):
            R = ts.node[state_label(s)]['rect'].copy()
            R[i][1] = R[i][0]
            if not is_facet_separating(system, R, -1, i, pert_bounds):
                ts.add_edge(state_label(s), state_label(index))
        R = ws.copy()
        R[i][0] = R[i][1]
        R[i][1] = np.infty
        index = [0 for j in range(system.n)]
        index[i] = len(partition[i]) - 1
        ts.add_node(state_label(index), rect=R, state=index, system=system)
        for s in it.product(*([list(range(len(p) - 1)) for p in partition[:i]] +
                              [[index[i] - 1]] +
                              [list(range(len(p) - 1)) for p in partition[i+1:]])):
            R = ts.node[state_label(s)]['rect'].copy()
            R[i][0] = R[i][1]
            if not is_facet_separating(system, R, 1, i, pert_bounds):
                ts.add_edge(state_label(s), state_label(index))

    return ts


