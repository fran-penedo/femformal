from subprocess import Popen, PIPE
import re
from os import path
import os
import tempfile

import logging
logger = logging.getLogger('FEMFORMAL')

NUSMV = path.join(path.dirname(__file__), 'nusmv', 'bin', 'NuSMV')

def check_spec(ts, spec, regions, init, verbose=False):
    f = tempfile.NamedTemporaryFile(prefix='nusmv_input', delete=False)
    f.write(ts.toNUSMV(spec, regions, init))
    f.close()
    process = [NUSMV, f.name]
    ps = Popen(process, stdout=PIPE, stderr=PIPE)
    out, err = ps.communicate()
    os.remove(f.name)
    if verbose:
        logger.debug(out)
        logger.debug(err)
    try:
        return _parse_nusmv(out, err)
    except ParserException:
        logger.debug(ts.toNUSMV(spec, regions, init))
        logger.debug(out)
        logger.debug(err)
        raise ParserException()

def _parse_nusmv(out, err):
    if out.find('true') != -1:
        return True, []
    elif err.find('Parser error') != -1:
        raise ParserException()
    else:
        lines = out.splitlines()
        try:
            start = next(i for i in range(len(lines))
                        if lines[i].startswith('Trace Type: Counterexample'))
            loop = next(i for i in range(len(lines))
                        if lines[i].startswith('  -- Loop starts here'))

            p = re.compile('state = s([0-9_]+)')
            matches = (p.search(line) for line in lines[start:])
            chain = [m.group(1) for m in matches if m is not None]
            if loop == len(lines) - 4:
                chain.append(chain[-1])

            trace = [(x, y) for x, y in zip(chain, chain[1:])]

            return False, trace
        except:
            raise ParserException()
class ParserException(Exception):

    def __init__(self):
        Exception.__init__(self)


