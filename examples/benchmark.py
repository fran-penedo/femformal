from __future__ import division, absolute_import, print_function

import argparse
import imp
import traceback
from timeit import default_timer as timer

import numpy as np

from femformal.core import system as sys, draw_util as draw, casestudy as fem
from femformal.core.logic import csystem_robustness
from femformal.femmilp.femmilp import verify_singleton, verify_set, synthesize
from femformal.femts.verify import verify, verify_input_constrained


def run_abstract(m, args):
    # logger.debug(m.system)
    start = timer()
    res = verify_input_constrained(
        m.system,
        m.partition,
        m.regions,
        m.init_states,
        m.spec,
        m.depth,
        draw_file_prefix=args.draw_file_prefix,
        verbosity=args.verbosity,
        draw_constr_ts=args.draw_constr_ts,
        check_inv=args.check_inv,
    )
    finish = timer()
    print("Res: {}".format(res))
    print("Time {}".format(finish - start))


def _get_gurobi_args(args):
    return {
        "threads": args.threads,
        "outputflag": args.outputflag,
        "numericfocus": args.numericfocus,
    }


def run_milp(m, args):
    # logger.debug(m.system)
    start = timer()
    if hasattr(m, "cs"):
        container = m.cs
    else:
        container = m
    res = verify_singleton(
        container.dsystem,
        container.d0,
        container.spec,
        container.fdt_mult,
        start_robustness_tree=getattr(container, "rob_tree", None),
        **_get_gurobi_args(args)
    )
    finish = timer()
    print("robustness = {}".format(res))
    print("time = {}".format(finish - start))


def run_milp_set(m, args):
    cs = m.cs
    start = timer()
    res = verify_set(
        cs.dsystem,
        cs.pset,
        cs.f,
        cs.spec,
        cs.fdt_mult,
        start_robustness_tree=getattr(cs, "rob_tree", None),
        **_get_gurobi_args(args)
    )
    finish = timer()
    print("robustness = {}".format(res))
    print("time = {}".format(finish - start))


def run_verify(m, args):
    cs = m.cs
    if "input_file" in args:
        inputm = load_module(args.input_file)
        # inputs = inputm.inputs
        # m.pwlf.ys = inputs
        draw_opts = draw.DrawOpts(getattr(inputm, "draw_opts", None))

        def f_nodal_control(t):
            f = np.zeros(m.N + 1)
            f[-1] = m.pwlf(t, x=m.pwlf.x)
            return f

        cs.system = sys.make_control_system(cs.system, f_nodal_control)
    res = csystem_robustness(cs.spec, cs.system, cs.d0)
    print("robustness = {}".format(res))


def run_milp_synth(m, args):
    cs = m.cs
    start = timer()
    res, synths = synthesize(
        cs.dsystem,
        cs.pset,
        cs.f,
        cs.spec,
        cs.fdt_mult,
        start_robustness_tree=getattr(cs, "rob_tree", None),
        **_get_gurobi_args(args)
    )
    finish = timer()
    print("robustness = {}".format(res))
    print("inputs = {}".format(synths))
    print("time = {}".format(finish - start))


def run_load(m, args):
    pass


def run_milp_batch(m, args):
    res = []
    times = []
    trues = []
    for cs, cstrue in zip(m.cslist, m.cstrues):
        print("---- cs")
        start = timer()
        res.append(verify_singleton(cs.dsystem, cs.d0, cs.spec))
        end = timer()
        times.append(end - start)
        trues.append(
            csystem_robustness(cstrue.spec, cstrue.system, cstrue.d0, cstrue.dt)
        )
        print("- time {}".format(times[-1]))
        print("- res {}".format(res[-1]))
        print("- trueres {}".format(trues[-1]))

    print("times: {}".format(times))
    print("results: {}".format(res))
    print("true results: {}".format(trues))


def run_abstract_batch(m, args):
    times = []
    its = 20
    for i in range(its):
        print("------ iteration {}".format(i))
        start = timer()
        _ = verify(
            m.system,
            m.partition,
            m.regions,
            m.init_states,
            m.spec,
            m.depth,
            plot_file_prefix=args.plot_file_prefix,
        )
        end = timer()
        times.append(end - start)
        print("- time {}".format(times[-1]))

    print(
        "verify times: max {0} min {1} avg {2}".format(
            max(times), min(times), sum(times) / float(its)
        )
    )


def run_draw(m, args):
    if "input_file" in args:
        inputm = load_module(args.input_file)
        draw_opts = draw.DrawOpts(getattr(inputm, "draw_opts", None))
    runstr = "run_draw_{}".format(args.draw_action)
    if runstr in globals():
        globals()[runstr](m, inputm, draw_opts, args)
    else:
        parser.print_help()


def run_draw_animated_2d(m, inputm, draw_opts, args):
    cs = getattr(m, "cs")
    cregions = getattr(m, "cregions", None)
    error_bounds = getattr(m, "error_bounds", None)

    if inputm is not None and hasattr(inputm, "inputs"):
        inputs = inputm.inputs
        m.pwlf.ys = inputs
        cs.system = sys.ControlSOSystem.from_sosys(
            cs.system, m.traction_force.traction_force
        )

    apcs, perts = _get_apcs_perts(
        cregions, error_bounds, cs.fdt_mult, mesh=cs.system.mesh
    )
    if draw_opts.deriv < 0:
        (fig,) = sys.draw_system_2d(
            cs.system,
            cs.d0,
            cs.g,
            cs.T,
            t0=0,
            hold=True,
            xlabel=draw_opts.xlabel,
            ylabel=draw_opts.ylabel,
            derivative_ylabel=draw_opts.derivative_ylabel,
        )
    else:
        (fig,) = sys.draw_system_deriv_2d(
            cs.system,
            cs.d0,
            cs.g,
            cs.T,
            draw_opts.deriv,
            t0=0,
            hold=True,
            xlabel=draw_opts.xlabel,
            ylabel=draw_opts.ylabel,
            derivative_ylabel=draw_opts.derivative_ylabel,
            apcs=apcs,
            labels=sorted(cregions.keys()),
            perts=perts,
            system_t=cs.system_t,
            d0_t=cs.d0_t,
        )

    # if cregions is not None:
    #     draw_predicates_to_axs_2d(fig.get_axes()[0:2], cregions, error_bounds,
    #                               cs.system.mesh, cs.fdt_mult)

    _set_fig_opts(fig, [0, 1], draw_opts, tight=False)

    draw.plt.show()


def run_draw_displacement(m, inputm, draw_opts, args):
    cs = getattr(m, "cs")
    cregions = getattr(m, "cregions", None)
    error_bounds = getattr(m, "error_bounds", None)

    if inputm is not None and hasattr(inputm, "inputs"):
        inputs = inputm.inputs
        m.pwlf.ys = inputs
        cs.system = sys.ControlSOSystem.from_sosys(
            cs.system, m.traction_force.traction_force
        )

    apcs, perts = _get_apcs_perts(
        cregions, error_bounds, cs.fdt_mult, mesh=cs.system.mesh
    )
    (fig,) = sys.draw_displacement_plot(
        cs.system,
        cs.d0,
        cs.g,
        cs.T,
        t0=0,
        hold=True,
        xlabel=draw_opts.xlabel,
        ylabel=draw_opts.ylabel,
        derivative_ylabel=draw_opts.derivative_ylabel,
        apcs=apcs,
        labels=sorted(cregions.keys()),
        perts=perts,
        system_t=cs.system_t,
        d0_t=cs.d0_t,
    )

    _set_fig_opts(fig, [0], draw_opts, tight=False)

    if args.movie:
        draw.save_ani(fig)
    draw.plt.show()


def run_draw_animated(m, inputm, draw_opts, args):
    cs = getattr(m, "cs")
    cregions = getattr(m, "cregions", None)
    error_bounds = getattr(m, "error_bounds", None)

    if inputm is not None and hasattr(inputm, "inputs"):
        inputs = inputm.inputs
        m.pwlf.ys = inputs

        def f_nodal_control(t):
            f = np.zeros(m.N + 1)
            f[-1] = m.pwlf(t, x=m.pwlf.x)
            return f

        cs.system = sys.make_control_system(cs.system, f_nodal_control)

    (fig,) = sys.draw_system(
        cs.system,
        cs.d0,
        cs.g,
        cs.T,
        t0=0,
        hold=True,
        xlabel=draw_opts.xlabel,
        ylabel=draw_opts.ylabel,
        derivative_ylabel=draw_opts.derivative_ylabel,
    )

    if cregions is not None:
        draw_predicates_to_axs(
            fig.get_axes()[0:2], cregions, error_bounds, cs.xpart, cs.fdt_mult
        )

    _set_fig_opts(fig, [0, 1, 2, 3], draw_opts, tight=False)

    draw.plt.show()


def run_draw_snapshots_disp(m, inputm, draw_opts, args):
    cs = getattr(m, "cs")
    cregions = getattr(m, "cregions", None)
    error_bounds = getattr(m, "error_bounds", None)
    if inputm is None:
        raise Exception("Input file needed to draw snapshots")
    ts = getattr(inputm, "ts")
    inputs = getattr(inputm, "inputs")
    if inputs is not None:
        m.pwlf.ys = inputs
        cs.system = sys.ControlSOSystem.from_sosys(
            cs.system, m.traction_force.traction_force
        )

    apcs, perts = _get_apcs_perts(
        cregions, error_bounds, cs.fdt_mult, mesh=cs.system.mesh
    )
    figs = sys.draw_displacement_snapshots(
        cs.system,
        cs.d0,
        cs.g,
        ts,
        xlabel=draw_opts.xlabel,
        ylabel=draw_opts.ylabel,
        derivative_ylabel=draw_opts.derivative_ylabel,
        apcs=apcs,
        labels=sorted(cregions.keys()),
        perts=perts,
        system_t=cs.system_t,
        d0_t=cs.d0_t,
    )

    ylims = [fig.get_axes()[0].get_ylim() for fig in figs]
    ylims_flat = [a for b in ylims for a in b]
    for fig, t in zip(figs, ts):
        _set_fig_opts(fig, [0], draw_opts, tight=False)
        draw.update_ax_ylim(fig.get_axes()[0], ylims_flat)

    prefix = draw_opts.file_prefix
    if prefix is None:
        draw.plt.show()
    else:
        for fig, t in zip(figs, ts):
            fig.savefig(
                _fix_filename(prefix + "_disp_t{}".format(t)) + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )


def run_draw_snapshots(m, inputm, draw_opts, args):
    cs = getattr(m, "cs")
    cregions = getattr(m, "cregions", None)
    error_bounds = getattr(m, "error_bounds", None)
    if inputm is None:
        raise Exception("Input file needed to draw snapshots")
    ts = getattr(inputm, "ts")
    inputs = getattr(inputm, "inputs")
    if inputs is not None:
        m.pwlf.ys = inputs

        def f_nodal_control(t):
            f = np.zeros(m.N + 1)
            f[-1] = m.pwlf(t, x=m.pwlf.x)
            return f

        cs.system = sys.make_control_system(cs.system, f_nodal_control)

    figs = sys.draw_system_snapshots(
        cs.system,
        cs.d0,
        cs.g,
        ts,
        hold=True,
        xlabel=draw_opts.xlabel,
        ylabel=draw_opts.ylabel,
        derivative_ylabel=draw_opts.derivative_ylabel,
        font_size=draw_opts.font_size,
    )
    figs_grouped = [(figs[2 * i], figs[2 * i + 1]) for i in range(len(ts))]

    for fig_pair, t in zip(figs_grouped, ts):
        if cregions is not None:
            draw_predicates_to_axs(
                [fig.get_axes()[0] for fig in fig_pair],
                cregions,
                error_bounds,
                cs.xpart,
                cs.fdt_mult,
            )
    for figs_group in zip(*figs_grouped):
        ylims = [fig.get_axes()[0].get_ylim() for fig in figs_group]
        ylims_flat = [a for b in ylims for a in b]
        for fig in figs_group:
            draw.update_ax_ylim(fig.get_axes()[0], ylims_flat)
            _set_fig_opts(fig, [0], draw_opts, tight=False)

    prefix = draw_opts.file_prefix
    if prefix is None:
        draw.plt.show()
    else:
        for fig_pair, t in zip(figs_grouped, ts):
            figa, figb = fig_pair
            figa.savefig(
                _fix_filename(prefix + "_disp_t{}".format(t)) + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )
            figb.savefig(
                _fix_filename(prefix + "_strain_t{}".format(t)) + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )


def run_draw_inputs(m, inputm, draw_opts, args):
    if inputm is None:
        raise Exception("Input file needed to draw snapshots")
    inputs = inputm.inputs
    m.pwlf.ys = inputs
    fig = sys.draw_pwlf(m.pwlf)

    ax = fig.get_axes()[0]
    ax.autoscale()
    ax.set_ylabel(draw_opts.input_ylabel)
    ax.set_xlabel(draw_opts.input_xlabel)
    _set_fig_opts(fig, [], draw_opts, tight=False)

    prefix = draw_opts.file_prefix
    if prefix is None:
        draw.plt.show()
    else:
        fig.savefig(
            _fix_filename(prefix + "_inputs") + ".png",
            bbox_inches="tight",
            pad_inches=0,
        )


def _set_fig_opts(fig, ax_indices, draw_opts, tight=True):
    fig.set_size_inches(draw_opts.plot_size_inches)
    fig.canvas.set_window_title(draw_opts.window_title)
    for i in ax_indices:
        ax = fig.get_axes()[i]
        ax.set_xticklabels(
            [
                (
                    ax.get_xticks()[j] * draw_opts.xaxis_scale
                    if j % draw_opts.xticklabels_pick == 0
                    else ""
                )
                for j in range(len(ax.get_xticks()))
            ]
        )
        ax.set_yticklabels(
            [
                (
                    ax.get_yticks()[j] * draw_opts.yaxis_scale
                    if j % draw_opts.yticklabels_pick == 0
                    else ""
                )
                for j in range(len(ax.get_yticks()))
            ]
        )
        # FIXME may break 1d figures, no idea why
        # draw.zoom_axes(ax, draw_opts.zoom_factors)
    for ax in fig.get_axes():
        try:
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
        except:
            pass
        try:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))
        except:
            pass
        for item in (
            [
                ax.title,
                ax.xaxis.label,
                ax.yaxis.label,
                ax.yaxis.get_offset_text(),
                ax.xaxis.get_offset_text(),
            ]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(draw_opts.font_size)
    if tight:
        fig.tight_layout()


def _fix_filename(s):
    return s.replace(".", "_")


def draw_predicates_to_axs(axs, cregions, error_bounds, xpart, fdt_mult):
    apcs, perts = _get_apcs_perts(cregions, error_bounds, fdt_mult, xpart=xpart)
    draw.draw_predicates(apcs, sorted(cregions.keys()), xpart, axs, perts=perts)


def draw_predicates_to_axs_2d(axs, cregions, error_bounds, mesh, fdt_mult):
    apcs, perts = _get_apcs_perts(cregions, error_bounds, fdt_mult, mesh=mesh)
    draw.draw_predicates_2d(apcs, sorted(cregions.keys()), mesh, axs, perts=perts)


def draw_predicates_to_axs_disp(ax, cregions, error_bounds, mesh, fdt_mult):
    draw.draw_predicates_displacement(
        apcs, sorted(cregions.keys()), mesh, ax, perts=perts
    )


def _get_apcs_perts(cregions, error_bounds, fdt_mult, xpart=None, mesh=None):
    apcs = zip(*sorted(cregions.items()))[1]
    if error_bounds is not None:
        epss, etas, nus = error_bounds
        try:
            perts = [
                fem.discretized_perturbed_profile(
                    apc, epss, etas, nus, xpart, fdt_mult, mesh
                )
                for apc in apcs
            ]
        except NotImplementedError:
            perts = [
                fem.perturb_profile(apc, epss, etas, nus, xpart, fdt_mult, mesh)[-1]
                for apc in apcs
            ]

    else:
        perts = None

    return apcs, perts


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="action")
    parser_draw = subparsers.add_parser("draw", help="Plot PDE trajectories")
    parser_draw.add_argument("-i", "--input-file", help="file with extra inputs")
    parser_draw.add_argument(
        "-m",
        "--movie",
        action="store_true",
        help="save a movie when drawing an animation",
    )
    _add_draw_argparser(parser_draw)
    parser_verify = subparsers.add_parser(
        "verify", help="Verify a model by direct integration"
    )
    parser_verify.add_argument("-i", "--input-file", help="file with extra inputs")
    parser_milp = subparsers.add_parser("milp", help="Run an example using MILP")
    parser_milp_set = subparsers.add_parser(
        "milp_set", help="Run an example for initial sets using MILP"
    )
    parser_milp_batch = subparsers.add_parser(
        "milp_batch", help="Run several examples in batch using MILP"
    )
    parser_milp_synth = subparsers.add_parser(
        "milp_synth", help="Run an example for synthesis using MILP"
    )
    parser_load = subparsers.add_parser("load", help="Load a benchmark file")
    parser.add_argument("module", help="file containing the case study")
    parser.add_argument(
        "--log-level", default="DEBUG", choices=["DEBUG", "INFO"], help="Logging level"
    )
    gurobi_group = parser.add_argument_group("Gurobi options")
    gurobi_group.add_argument(
        "--gthreads",
        dest="threads",
        type=int,
        default=10,
        help="Number of threads that gurobi will use",
    )
    gurobi_group.add_argument(
        "--goutputflag",
        dest="outputflag",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable (1) or disable (0) gurobi output",
    )
    gurobi_group.add_argument(
        "--gnumericfocus",
        dest="numericfocus",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Degree of control over numerical issues",
    )
    return parser


def _add_draw_argparser(parser):
    subparsers = parser.add_subparsers(dest="draw_action")
    subparsers.add_parser("animated")
    subparsers.add_parser("animated_2d")
    subparsers.add_parser("inputs")
    subparsers.add_parser("snapshots")
    subparsers.add_parser("snapshots_disp")
    subparsers.add_parser("displacement")


def load_module(f):
    try:
        return imp.load_source("femformal_benchmark_module", f)
    except Exception:
        print("Couldn't load module {}".format(f))
        traceback.print_exc()
        return None


def _logging_setup(args):
    import logging.config

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "debug_formatter": {
                    "format": "%(levelname).1s %(module)s:%(lineno)d:%(funcName)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "debug_formatter",
                }
            },
            "loggers": {
                "femformal": {
                    "handlers": ["console"],
                    "level": args.log_level,
                    "propagate": True,
                },
                "stlmilp": {
                    "handlers": ["console"],
                    "level": args.log_level,
                    "propagate": True,
                },
                "py.warnings": {
                    "handlers": ["console"],
                    "level": args.log_level,
                    "propagate": True,
                },
            },
        }
    )
    logging.captureWarnings(True)


def main():
    parser = get_argparser()
    args = parser.parse_args()
    _logging_setup(args)

    runstr = "run_" + args.action
    module = load_module(args.module)
    if module is None:
        exit()
    if runstr in globals():
        globals()[runstr](module, args)
    else:
        parser.print_help()
