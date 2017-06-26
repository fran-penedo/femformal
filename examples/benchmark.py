import argparse
import numpy as np
import imp
import logging
from timeit import default_timer as timer

import femformal.core.system as sys
import femformal.core.draw_util as draw
import femformal.core.fem.fem_util as fem
from femformal.core.logic import csystem_robustness
from femformal.femmilp.femmilp import verify_singleton, verify_set, synthesize
from femformal.femts.verify import verify, verify_input_constrained


logger = logging.getLogger('FEMFORMAL')

def run_abstract(m, args):
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

def run_milp(m, args):
    # logger.debug(m.system)
    start = timer()
    res = verify_singleton(m.dsystem, m.d0, m.spec, m.fdt_mult)
    finish = timer()
    print 'Res: {}'.format(res)
    print 'Time {}'.format(finish - start)

def run_milp_set(m, args):
    cs = m.cs
    start = timer()
    res = verify_set(cs.dsystem, cs.pset, cs.f, cs.spec, cs.fdt_mult)
    finish = timer()
    print 'Res: {}'.format(res)
    print 'Time {}'.format(finish - start)

def run_milp_synth(m, args):
    cs = m.cs
    start = timer()
    res, synths = synthesize(cs.dsystem, cs.pset, cs.f, cs.spec, cs.fdt_mult)
    finish = timer()
    print 'robustness = {}'.format(res)
    print 'inputs = {}'.format(synths)
    print 'time = {}'.format(finish - start)

def run_load(m, args):
    pass

def run_milp_batch(m, args):
    res = []
    times = []
    trues = []
    for cs, cstrue in zip(m.cslist, m.cstrues):
        print "---- cs"
        start = timer()
        res.append(verify_singleton(cs.dsystem, cs.d0, cs.spec))
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

def run_abstract_batch(m, args):
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

def run_draw(m, args):
    runstr = 'run_draw_{}'.format(args.draw_action)
    if runstr in globals():
        globals()[runstr](m, args)
    else:
        parser.print_help()

def run_draw_animated(m, args):
    cs = getattr(m, 'cs')
    cregions = getattr(m, 'cregions', None)
    error_bounds = getattr(m, 'error_bounds', None)

    if 'input_file' in args:
        inputm = load_module(args.input_file)
        inputs = inputm.inputs
        m.pwlf.ys = inputs
        def f_nodal_control(t):
            f = np.zeros(m.N + 1)
            f[-1] = m.pwlf(t, x=m.pwlf.x)
            return f
        cs.system = sys.ControlSOSystem.from_sosys(cs.system, f_nodal_control)

    (fig, ) = sys.draw_system(cs.system, cs.d0, cs.g, cs.T, t0=0, hold=True)

    if cregions is not None:
        draw_predicates_to_axs(fig.get_axes()[0:2], cregions, error_bounds, cs.xpart, cs.fdt_mult)

    fig.set_size_inches(3,2)
    fig.canvas.set_window_title("i3_7")

    draw.plt.show()

def run_draw_snapshots(m, args):
    cs = getattr(m, 'cs')
    cregions = getattr(m, 'cregions', None)
    error_bounds = getattr(m, 'error_bounds', None)
    inputm = load_module(args.input_file)
    ts = getattr(inputm, 'ts')
    inputs = getattr(inputm, 'inputs')
    if inputs is not None:
        m.pwlf.ys = inputs
        def f_nodal_control(t):
            f = np.zeros(m.N + 1)
            f[-1] = m.pwlf(t, x=m.pwlf.x)
            return f
        cs.system = sys.ControlSOSystem.from_sosys(cs.system, f_nodal_control)

    figs = sys.draw_system_snapshots(cs.system, cs.d0, cs.g, ts, hold=True)
    figs_grouped = [(figs[2*i], figs[2*i + 1]) for i in range(len(ts))]

    for fig_pair, t in zip(figs_grouped, ts):
        for fig in fig_pair:
            fig.set_size_inches(3,2)
            fig.canvas.set_window_title("i3_7")
        if cregions is not None:
            draw_predicates_to_axs(
                [fig.get_axes()[0] for fig in fig_pair], cregions,
                error_bounds, cs.xpart, cs.fdt_mult)
    for figs_group in zip(*figs_grouped):
        ylims = [fig.get_axes()[0].get_ylim() for fig in figs_group]
        ylims_flat = [a for b in ylims for a in b]
        for fig in figs_group:
            draw.update_ax_ylim(fig.get_axes()[0], ylims_flat)

    prefix = args.draw_file_prefix
    if prefix is None:
        draw.plt.show()
    else:
        for fig_pair, t in zip(figs_grouped, ts):
            figa, figb = fig_pair
            figa.savefig(_fix_filename(prefix + "_disp_t{}".format(t)) + ".png")
            figb.savefig(_fix_filename(prefix + "_strain_t{}".format(t)) + ".png")

def _fix_filename(s):
    return s.replace('.', '_')

def run_draw_inputs(m, args):
    inputm = load_module(args.input_file)
    inputs = inputm.inputs
    m.pwlf.ys = inputs
    fig = sys.draw_pwlf(m.pwlf)

    fig.set_size_inches(3,2)
    fig.canvas.set_window_title("i3_7")
    fig.get_axes()[0].autoscale()

    prefix = args.draw_file_prefix
    if prefix is None:
        draw.plt.show()
    else:
        fig.savefig(_fix_filename(prefix + "_inputs") + ".png")

def draw_predicates_to_axs(axs, cregions, error_bounds, xpart, fdt_mult):
    apcs = zip(*sorted(cregions.items()))[1]
    if error_bounds is not None:
        epss, etas, nus = error_bounds
        perts = [fem.perturb_profile(apc, epss, etas, nus, xpart, fdt_mult)
                for apc in apcs]
    else:
        perts = None
    draw.draw_predicates(apcs, sorted(cregions.keys()), xpart, axs, perts=perts)


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='action')
    parser_abstract = subparsers.add_parser(
        'abstract', help='Run an example once and plot evolution')
    parser_abstract.add_argument('-f', '--draw-file-prefix',
                             help='plots are saved to svg files with this prefix')
    parser_abstract.add_argument('--draw-constr-ts', action='store_true')
    parser_abstract_batch = subparsers.add_parser(
        'abstract_batch', help='Run an example a number of times a show execution times')
    parser_draw = subparsers.add_parser(
        'draw', help='Plot PDE trajectories')
    parser_draw.add_argument('-f', '--draw-file-prefix',
                             help='save plots to files with this prefix')
    parser_draw.add_argument('-i', '--input-file', help='file with extra inputs')
    _add_draw_argparser(parser_draw)
    parser_milp = subparsers.add_parser(
        'milp', help='Run an example using MILP')
    parser_milp_set = subparsers.add_parser(
        'milp_set', help='Run an example for initial sets using MILP')
    parser_milp_batch = subparsers.add_parser(
        'milp_batch', help='Run several examples in batch using MILP')
    parser_milp_synth = subparsers.add_parser(
        'milp_synth', help='Run an example for synthesis using MILP')
    parser_load = subparsers.add_parser(
        'load', help='Load a benchmark file')
    parser.add_argument('module', help='file containing the case study')
    parser.add_argument('-v', '--verbosity', action='count')
    parser.add_argument('--check-inv', action='store_true')
    return parser

def _add_draw_argparser(parser):
    subparsers = parser.add_subparsers(dest='draw_action')
    subparsers.add_parser('animated')
    subparsers.add_parser('inputs')
    subparsers.add_parser('snapshots')


def load_module(f):
    try:
        return imp.load_source("femformal_benchmark_module", f)
    except Exception as e:
        print "Couldn't load module {}".format(f)
        print e
        return None

def main():
    parser = get_argparser()
    args = parser.parse_args()
    runstr = 'run_' + args.action
    module = load_module(args.module)
    if module is None:
        exit()
    if runstr in globals():
        globals()[runstr](module, args)
    else:
        parser.print_help()
