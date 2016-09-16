from femformal.verify import verify
import femformal.util as util
import numpy as np
import argparse
from timeit import default_timer as timer
import importlib

import cProfile

def run_cs_draw(m, args):
    start = timer()
    res = verify(m.system, m.partition, m.regions, m.init_states, m.spec, m.depth,
                   plot_file_prefix=args.plot_file_prefix)
    finish = timer()
    print 'Time {}'.format(finish - start)


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
    parser_draw.add_argument('-f', '--plot-file-prefix',
                             help='plots are saved to svg files with this prefix')
    parser_time = subparsers.add_parser(
        'time', help='Run an example a number of times a show execution times')
    parser.add_argument('module', help='module containing the case study')
    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    module = importlib.import_module(args.module)
    if args.action == 'draw':
        # command = "run_cs_draw(module, args)"
        # cProfile.runctx(command, globals(), locals(), filename="prof.profile")
        run_cs_draw(module, args)
    elif args.action == 'time':
        run_cs_time(module, args)
    else:
        parser.print_help()
