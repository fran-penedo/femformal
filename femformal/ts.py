import networkx as nx

from os import path

NUSMV = path.join(path.dirname(__file__), 'nusmv', 'NuSMV')

class TS(nx.DiGraph):

    def __init__(self):
        nx.Graph.__init__(self)

    def modelcheck(self):
        ps = Popen(NUSMV, stdin=PIPE, stdout=PIPE)
        out = ps.communicate(self.toNUSMV())[0]
        try:
            return parse_nusmv(out)
        except:
            print self.toNUSMV()
            raise Exception()

    def toNUSMV(self):
        out = StringIO()
        print >>out, 'MODULE main'
        print >>out, 'VAR'
        print >>out, 'state : {};'.format(nusmv_statelist(self.nodes()))
        print >>out, 'ASSIGN'
        print >>out, 'init(state) := {};'.format(
            nusmv_statelist([self.nodes()[i] for i in self._init]))
        print >>out, 'next(state) := '
        print >>out, 'case'

        for node in self.nodes():
            succ = G.successors(node)
            if len(succ) > 0:
                print >>out, 'state = {} : {};'.format(
                    "s" + node,
                    nusmv_statelist(succ))

        print >>out, 'TRUE : state;'
        print >>out, 'esac;'
        if self._ltl is not None:
            print >>out, 'LTLSPEC {}'.format(self._ltl)

        s = out.getvalue()
        out.close()
        return s

def nusmv_statelist(l):
    return '\{{}\}'.format(', '.join(map(lambda x: "s" + x, l)))

def parse_nusmv(out):
    if out.find('true') != -1:
        return True, []
    elif out.find('Parser error') != -1:
        print out
        raise Exception()
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

