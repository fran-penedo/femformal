from femformal.verify import verify, verify_input_constrained
from femmilp.system_milp import rh_system_sat, csystem_robustness
import femformal.util as util
import numpy as np
import argparse
from timeit import default_timer as timer
import importlib

import cProfile

import logging
logger = logging.getLogger('FEMFORMAL')

def run_cs_draw(m, args):
    logger.debug(m.system)
    start = timer()
    res = verify_input_constrained(m.system, m.partition, m.regions, m.init_states, m.spec, m.depth,
                 draw_file_prefix=args.draw_file_prefix,
                 verbosity=args.verbosity,
                 draw_constr_ts=args.draw_constr_ts,
                 check_inv=args.check_inv)
    finish = timer()
    print 'Res: {}'.format(res)
    print 'Time {}'.format(finish - start)

def run_cs_milp(m, args):
    logger.debug(m.system)
    start = timer()
    res = rh_system_sat(m.system, m.d0, m.rh_N, m.spec)
    finish = timer()
    print 'Res: {}'.format(res)
    print 'Time {}'.format(finish - start)

def run_load(m, args):
    pass

def run_cs_milp_batch(m, args):
    res = []
    times = []
    trues = []
    for cs, cstrue in zip(m.cslist, m.cstrues):
        print "---- cs"
        start = timer()
        res.append(rh_system_sat(cs.system, cs.d0, cs.rh_N, cs.spec))
        end = timer()
        times.append(end - start)
        trues.append(csystem_robustness(cstrue.spec, cstrue.system,
                                        cstrue.d0, cstrue.dt))
        print "- time {}".format(times[-1])
        print "- res {}".format(res[-1])
        print "- trueres {}".format(trues[-1])

    print "times: {}".format(times)
    print "results: {}".format(res)
    print "true results: {}".format(trues)

def run_cs_time(m, args):
    times = []
    its = 20
    for i in range(its):
        print "------ iteration {}".format(i)
        start = timer()
        _ = verify(m.system, m.partition, m.regions, m.init_states, m.spec,
                   m.depth, plot_file_prefix=args.plot_file_prefix)
        end = timer()
        times.append(end - start)
        print "- time {}".format(times[-1])

    print "verify times: max {0} min {1} avg {2}".format(
        max(times), min(times), sum(times) / float(its))


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='action')
    parser_draw = subparsers.add_parser(
        'draw', help='Run an example once and plot evolution')
    parser_draw.add_argument('-f', '--draw-file-prefix',
                             help='plots are saved to svg files with this prefix')
    parser_draw.add_argument('--draw-constr-ts', action='store_true')
    parser_time = subparsers.add_parser(
        'time', help='Run an example a number of times a show execution times')
    parser_milp = subparsers.add_parser(
        'milp', help='Run an example using MILP')
    parser_milp_batch = subparsers.add_parser(
        'milp_batch', help='Run several examples in batch using MILP')
    parser_load = subparsers.add_parser(
        'load', help='Load a benchmark file')
    parser.add_argument('module', help='module containing the case study')
    parser.add_argument('-v', '--verbosity', action='count')
    parser.add_argument('--check-inv', action='store_true')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()
    module = importlib.import_module(args.module)
    if args.action == 'draw':
        # command = "run_cs_draw(module, args)"
        # cProfile.runctx(command, globals(), locals(), filename="prof.profile")
        run_cs_draw(module, args)
    elif args.action == 'time':
        run_cs_time(module, args)
    elif args.action == 'milp':
        run_cs_milp(module, args)
    elif args.action == 'milp_batch':
        run_cs_milp_batch(module, args)
    elif args.action == 'load':
        run_load(module, args)
    else:
        parser.print_help()
