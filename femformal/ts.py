import networkx as nx
import numpy as np
import itertools as it
from subprocess import Popen, PIPE
import re
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
from femformal.util import state_label, subst_spec_labels
from femformal.system import is_facet_separating

import logging
logger = logging.getLogger('FEMFORMAL')

from os import path

NUSMV = path.join(path.dirname(__file__), 'nusmv', 'NuSMV')

class TS(nx.DiGraph):

    def __init__(self):
        nx.DiGraph.__init__(self)

    def add_transition(self, x, y, pert_bounds):
        xs, ys = label_state(x), label_state(y)
        xs, ys = long_first(xs, ys)
        xn = self.node[xs]
        dim, normal = first_change(xs, ys)

        R = xn['rect'].copy()
        if normal == -1:
            R[dim][1] = R[dim][0]
        else:
            R[dim][0] = R[dim][1]
        if not is_facet_separating(xn['system'], R, normal, dim, pert_bounds):
            ts.add_edge(state_label(xs), state_label(ys))


    def modelcheck(self, spec, regions, init):
        ps = Popen(NUSMV, stdin=PIPE, stdout=PIPE)
        out = ps.communicate(self.toNUSMV(spec, regions, init))[0]
        try:
            return _parse_nusmv(out)
        except ParserException:
            print self.toNUSMV(spec, regions, init)
            raise ParserExceptionException()

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


def abstract(system, partition, pert_bounds):
    if len(partition) != system.m + system.n:
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


def _nusmv_statelist(l):
    return '{{{}}}'.format(', '.join(l))

def _parse_nusmv(out):
    if out.find('true') != -1:
        return True, []
    elif out.find('Parser error') != -1:
        print out
        raise ParserException()
    else:
        lines = out.splitlines()
        start = next(i for i in range(len(lines))
                     if lines[i].startswith('Trace Type: Counterexample'))
        loop = next(i for i in range(len(lines))
                    if lines[i].startswith('-- Loop starts here'))

        p = re.compile('state = s([0,1]+)')
        matches = (p.search(line) for line in lines[start:])
        chain = [m.group(1) for m in matches if m is not None]
        if loop == len(lines) - 4:
            chain.append(chain[-1])

        trace = [(x, y) for x, y in zip(chain, chain[1:])]

        return False, trace

class ParserException(Exception):

    def __init__(self):
        Exception.__init__(self)


