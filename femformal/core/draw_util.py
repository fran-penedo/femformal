from bisect import bisect_left, bisect_right
import logging

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import animation as animation, cm as cmx, colors as colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d as p3


logger = logging.getLogger(__name__)

_figcounter = 0
_holds = []

def draw_linear(ys, xs, ylabel='y', xlabel='x', axes=None):
    matplotlib.rcParams.update({'font.size': 8})
    matplotlib.rcParams.update({'figure.autolayout': True})
    if axes is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes
    # ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #FIXME
    ax.set_ylim(-100, 6e3)
    ax.plot(xs, ys, '-')

    if axes is None:
        return fig

def draw_ts(ts, prefix=None):
    global _figcounter

    nx.draw_networkx(ts)

    if prefix is not None:
        plt.savefig(prefix + str(_figcounter) + '.svg')
        plt.show()
        _figcounter += 1
    else:
        plt.show()

def draw_ts_2D(ts, partition, prefix=None):
    global _figcounter

    if len(label_state(ts.nodes()[0])) != 2:
        raise ValueError("Expected TS from 2D partition")
    if len(partition) != 2:
        raise ValueError("Expected 2D partition")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _draw_grid(partition, ax)
    for e in ts.edges():
        _draw_edge(e, partition, ax)

    if prefix is not None:
        fig.savefig(prefix + str(_figcounter) + '.svg')
        plt.close(fig)
        plt.show()
        _figcounter += 1
    else:
        plt.show()


def draw_pde_trajectories(dss, xss, tss, pwc=False, ylabel="u", xlabel="x", derivative_ylabel="$\\frac{d}{dx} u$"):
    global _holds

    d_min, d_max = np.amin([np.amin(ds) for ds in dss]), np.amax([np.amax(ds) for ds in dss])
    x_min, x_max = np.amin(np.hstack(xss)), np.amax(np.hstack(xss))
    t_finer_index = np.argmin([ts[1] - ts[0] for ts in tss])
    ts = tss[t_finer_index]

    matplotlib.rcParams.update({'font.size': 8})
    matplotlib.rcParams.update({'figure.autolayout': True})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(d_min, d_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    time_text = ax.text(.02, .95, '', transform=ax.transAxes)

    cmap = plt.get_cmap('autumn')
    cnorm = colors.Normalize(ts[0], ts[-1])
    scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    scalarmap.set_array(ts)
    if pwc:
        ls = []
        for ds in dss:
            l = ax.add_collection(LineCollection([], lw=3))
            ls.append(l)
        def update_line(i):
            next_t = ts[i]
            for j in range(len(ls)):
                ii = bisect_right(tss[j], next_t) - 1
                ls[j].set_segments([np.array([[xss[j][k], dss[j][ii][k]],
                                          [xss[j][k+1], dss[j][ii][k]]])
                                for k in range(len(xss[j]) - 1)])
                # ls[j].set_color(scalarmap.to_rgba(next_t))
                ls[j].set_color(['blue','red','green'][j])
            time_text.set_text('t = {}'.format(ts[i]))
            return tuple(ls) + (time_text,)

    else:
        ls = []
        for ds in dss:
            l, = ax.plot([], [], 'b-')
            ls.append(l)
        def update_line(i):
            next_t = ts[i]
            for j in range(len(ls)):
                k = bisect_right(tss[j], next_t) - 1
                ls[j].set_data(xss[j], dss[j][k])
                # ls[j].set_color(scalarmap.to_rgba(next_t))
                ls[j].set_color(['blue','red','green'][j])
            time_text.set_text('t = {}'.format(ts[i]))
            return tuple(ls) + (time_text,)

    frames = len(ts)
    line_ani = animation.FuncAnimation(
        fig, update_line, frames=frames, interval=5000/frames, blit=True)

    def onClick(event):
        if onClick.anim_running:
            line_ani.event_source.stop()
            onClick.anim_running = False
        else:
            line_ani.event_source.start()
            onClick.anim_running = True
    onClick.anim_running = True
    fig.canvas.mpl_connect('button_press_event', onClick)
    _holds.append(line_ani)

    _render(fig, None, False)


def _der_lines(ax, xs):
    return [ax.plot([], [], 'b-')[0] for x in xs[:-1]]

def _combine_lines(lines):
    return lambda i: sum([tuple(line(i)) for line in lines], ())

def set_traj_line(ax, ds, xs, ts, hlines=False, xlabel='x', ylabel='u', scalarmap=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xs[0], xs[-1])
    d_min, d_max = np.amin(ds), np.amax(ds)
    ax.set_ylim(d_min, d_max)
    time_text = ax.text(.02, .95, '', transform=ax.transAxes)
    # if scalarmap is None:
    #     cmap = plt.get_cmap('autumn')
    #     cnorm = colors.Normalize(ts[0], ts[-1])
    #     scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    #     scalarmap.set_array(ts)

    if hlines:
        l = ax.add_collection(LineCollection([]))
        # _der_lines(ax, xs)
        def update_line(i, l=l):
            l.set_segments([np.array([[xs[j], ds[i][j]], [xs[j+1], ds[i][j]]])
                            for j in range(len(xs) - 1)])
            # l[j].set_data(xs[j:j+2], [ds[i][j], ds[i][j]])
            if scalarmap is not None:
                l.set_color(scalarmap.to_rgba(ts[i]))
            else:
                l.set_color('black')
            time_text.set_text('t = {}'.format(ts[i]))
            return l, time_text


    else:
        l, = ax.plot([], [], 'b-')

        def update_line(i, l=l):
            l.set_data(xs, ds[i])
            if scalarmap is not None:
                l.set_color(scalarmap.to_rgba(ts[i]))
            else:
                l.set_color('black')
            time_text.set_text('t = {}'.format(ts[i]))
            return l, time_text


    return update_line

def set_animation(fig, ts, lines):
    fig.subplots_adjust(bottom=0.1)
    axslider = plt.axes([0.0, 0.0, 1, .03])
    slider = Slider(axslider, 'Time', ts[0], ts[-1], valinit=ts[0])

    frames = len(ts)
    def frame_seq():
        while True:
            frame_seq.cur = (frame_seq.cur + 1) % frames
            # slider.set_val(ts[frame_seq.cur])
            yield frame_seq.cur
    frame_seq.cur = 0

    def ani_func(i):
        return lines(i)

    line_ani = animation.FuncAnimation(fig, ani_func, frames=frame_seq,
        interval=max(5000/frames, 20), blit=False)

    def onClick(event):
        if event.inaxes == axslider: return

        if onClick.anim_running:
            fig.__line_ani.event_source.stop()
            onClick.anim_running = False
        else:
            fig.__line_ani.event_source.start()
            onClick.anim_running = True

    onClick.anim_running = True
    fig.canvas.mpl_connect('button_press_event', onClick)

    def onChanged(val):
        t = bisect_left(ts, val)
        frame_seq.cur = t
        line_ani = animation.FuncAnimation(fig, ani_func, frames=frame_seq,
            interval=max(5000/frames, 20), blit=False)
        fig.__line_ani = line_ani

    slider.on_changed(onChanged)
    slider.drawon = False
    # lines(0)
    fig.__slider = slider
    fig.__line_ani = line_ani

    return line_ani


def draw_pde_trajectory(ds, xs, ts, hold=False,
                        ylabel='Temperature', xlabel='x', derivative_ylabel="$\\frac{d}{dx} Temperature$"):
    global _holds

    matplotlib.rcParams.update({'font.size': 8})
    # matplotlib.rcParams.update({'figure.autolayout': True})
    cmap = plt.get_cmap('autumn')
    cnorm = colors.Normalize(ts[0], ts[-1])
    scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    scalarmap.set_array(ts)

    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
    ((ax_tl, ax_tr), (ax_bl, ax_br)) = axes
    line_tl = set_traj_line(ax_tl, ds, xs, ts, xlabel=xlabel, ylabel=ylabel,
                            scalarmap=None)
    line_bl = set_traj_line(ax_bl, ds, xs, ts, xlabel=xlabel, ylabel=ylabel,
                            scalarmap=scalarmap)
    dds = np.true_divide(np.diff(ds), np.diff(xs))
    line_tr = set_traj_line(ax_tr, dds, xs, ts, hlines=True, xlabel=xlabel,
                            ylabel=derivative_ylabel, scalarmap=None)
    line_br = set_traj_line(ax_br, dds, xs, ts, hlines=True, xlabel=xlabel,
                            ylabel=derivative_ylabel, scalarmap=scalarmap)

    savefun = None

    lines = _combine_lines([line_tl, line_tr])
    set_animation(fig, ts, lines)

    for i in range(len(ts)):
        line_bl(i, l=ax_bl.plot([], [], 'b-')[0])
        # line_br(i, l=_der_lines(ax_br, xs))
        line_br(i, l=ax_br.add_collection(LineCollection([])))
    # fig.subplots_adjust(right=0.5)
    # axcbar = fig.add_axes([.85, 0.15, 0.05, 0.7])ax=axes.ravel().tolist(),
    cbar = fig.colorbar(scalarmap, use_gridspec=True)
    cbar.set_label('Time t (s)')
    for ax in axes.flatten():
        zoom_axes(ax, .05)
    fig.tight_layout()

    _render(fig, savefun, hold)


def draw_2d_pde_trajectory(ds, nodes_coords, elems_nodes, ts, **kwargs):
    global _holds
    fig = plt.figure()
    axl = fig.add_subplot(121, projection='3d')
    axr = fig.add_subplot(122, projection='3d')
    axes = [axl, axr]

    poly_update_fs = []
    polys_verts = np.array([_polygon_vertices_2dof(nodes_coords, d, elems_nodes)
                            for d in ds])
    for i in range(2):
        poly_update = set_traj_poly(axes[i], polys_verts[:, i], ts)
        poly_update_fs.append(poly_update)

    for i, ax in enumerate(axes):
        ax.set_xlim3d(np.amin(polys_verts[:, i, :, :, 0]), np.amax(polys_verts[:, i, :, :, 0]))
        ax.set_ylim3d(np.amin(polys_verts[:, i, :, :, 1]), np.amax(polys_verts[:, i, :, :, 1]))
        ax.set_zlim3d(np.amin(polys_verts[:, i, :, :, 2]), np.amax(polys_verts[:, i, :, :, 2]))

    set_animation(fig, ts, _combine_lines(poly_update_fs))

    fig.tight_layout()
    _holds.append(fig)

def set_traj_poly(ax, verts, ts):
    time_text = ax.text2D(.02, .95, '', transform=ax.transAxes)
    poly_collection = ax.add_collection(p3.art3d.Poly3DCollection([]))

    def update_line(i, p=poly_collection):
        p.set_verts(verts[i])
        time_text.set_text('t = {}'.format(ts[i]))
        return p, time_text

    return update_line

def _polygon_vertices_2dof(nodes_coords, d, elems_nodes):
    return np.array([_polygon_vertices(nodes_coords, d[i::2], elems_nodes)
                     for i in range(2)])

def _polygon_vertices(nodes_coords, d, elems_nodes):
    verts = np.array([
        [np.hstack([nodes_coords[n], d[n]]) for n in elem_nodes]
        for elem_nodes in elems_nodes])
    return verts

def _set_snap_figure(t, xlabel, ylabel, font_size, ylims=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.text(.02, .90, 't = {}'.format(t), transform=ax.transAxes, fontsize=font_size)
    return fig

def draw_pde_snapshot(
    xs, ds, dds, t, ylabel='u', xlabel='x', derivative_ylabel="$\\frac{d}{dx} u$",
    font_size=8, ylims=None, hold=False):
    fig = _set_snap_figure(t, xlabel, ylabel, font_size, ylims)
    ax = fig.get_axes()[0]
    ax.plot(xs, ds, 'k-')
    zoom_axes(ax, .05)

    _render(fig, None, hold)

    fig = _set_snap_figure(t, xlabel, derivative_ylabel, font_size, ylims)
    ax = fig.get_axes()[0]
    ax.hlines(dds, xs[:-1], xs[1:], colors='k')
    zoom_axes(ax, .05)

    _render(fig, None, hold)


def draw_displacement_2d(ds, mesh, ts, apcs=None, labels=None, perts=None, **kwargs):
    global _holds
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim(mesh.partitions[0][0], mesh.partitions[0][-1])
    ax.set_xticks(mesh.partitions[0])
    ax.set_ylim(mesh.partitions[1][0], mesh.partitions[1][-1])
    ax.set_yticks(mesh.partitions[1])
    zoom_axes(ax, .9)
    ax.grid()

    ls = ax.add_collection(LineCollection([], colors='k'))
    time_text = ax.text(.02, .95, '', transform=ax.transAxes)

    def lines(i):
        nodes = mesh.nodes_coords + ds[i].reshape(mesh.nodes_coords.shape[0], 2)
        ls.set_segments([[nodes[n], nodes[nex]]
                         for n in range(mesh.nnodes)
                         for nex in mesh.connected_fwd(n)])
        time_text.set_text('t = {}'.format(ts[i]))
        return ls, time_text


    pred_data = predicates_displacement_data(apcs, labels, mesh, ax, perts)
    pred_lines = [pred_data[i]['l'] for i in pred_data.keys()] + \
        [pred_data[i]['l_pert']
         for i in pred_data.keys() if 'l_pert' in pred_data[i]]
    def preds(i):
        for key, value in pred_data.items():
            dof = apcs[key].u_comp
            data = value['pts'].copy()
            data[:,dof] += value['disps']
            sys_disps = np.array([ds[i][2*n + 1 - dof] for n in value['nodes']])
            data[:, 1 - dof] += sys_disps
            value['l'].set_data(data[:,0], data[:,1])
            if 'l_pert' in value:
                interp = mesh.interpolate(ds[i])
                pert_data = value['pts_pert'].copy()
                pert_sys_disps = np.array([interp(*c)[1 - dof] for c in pert_data])
                pert_data[:,dof] += value['disps_pert']
                pert_data[:, 1 - dof] += pert_sys_disps
                value['l_pert'].set_data(pert_data[:,0], pert_data[:,1])
        return tuple(pred_lines)


    ax.legend(loc='lower left', fontsize='6', labelspacing=0.05, handletextpad=0.1)

    set_animation(fig, ts, _combine_lines([lines, preds]))
    fig.tight_layout()

    _render(fig, None, kwargs['hold'])


def _render(fig, savefun, hold):
    global _holds
    global _figcounter

    if not hold:
        if savefun is not None:
            savefun()
            plt.close(fig)
            plt.show()
        else:
            plt.show()
        _holds = []

    _holds.append(fig)
    _figcounter += 1

def pop_holds():
    global _holds
    h = _holds
    _holds = []
    return h

def _draw_edge(e, partition, ax):
    f = label_state(e[0])
    t = label_state(e[1])
    i, d = first_change(f, t)

    fcenter, fsize = _box_dims(partition, f)
    tcenter, tsize = _box_dims(partition, t)

    if d != 0:
        fcenter += d * fsize / 4.0
        tcenter += d * fsize / 4.0
        tcenter[i] -= 2 * d * fsize[i] / 4.0
        ax.arrow(*np.hstack([fcenter, (tcenter - fcenter)]), color='b', width=.01,
                 head_width=.1, head_starts_at_zero=False)
    else:
        ax.plot(*fcenter, color='b', marker='x', markersize=10)

def _box_dims(partition, state):
    bounds = np.array([[partition[i][s], partition[i][s+1]]
                       for i, s in enumerate(state)])
    return np.mean(bounds, axis=1), bounds[:,1] - bounds[:,0]

def _draw_grid(partition, ax):
    limits = np.array([[p[0], p[-1]] for p in partition])
    for i, p in enumerate(partition):
        data = limits.copy()
        for x in p:
            data[i][0] = data[i][1] = x
            ax.plot(data[0], data[1], color='k', linestyle='--', linewidth=2)


def draw_predicates(apcs, labels, xpart, axs, perts=None):
    if perts is not None:
        mids = (xpart[:-1] + xpart[1:]) / 2.0

    for i, apc in enumerate(apcs):
        ax = axs[apc.uderivs]
        ys = [apc.p(x) for x in apc.A]
        if not np.isclose(apc.A[0], apc.A[1]):
            (l,) = ax.plot(apc.A, ys, lw=1, label=labels[i])
        else:
            (l,) = ax.plot(apc.A[0], ys[0], marker='_', markersize=3, label=labels[i])
        update_ax_ylim(ax, ys)
        if perts is not None:
            mids_in_domain = [x for x in mids if x >= apc.A[0] and x <= apc.A[1]]
            if len(mids_in_domain) > 1:
                ys = [perts[i](x) for x in mids_in_domain]
                ax.plot(mids_in_domain, ys, lw=1, ls='--', c=l.get_c())
            elif len(mids_in_domain) == 1:
                ys = [perts[i](x) for x in mids_in_domain]
                ax.plot(mids_in_domain, ys, marker='_', markersize=3, c=l.get_c())
            else:
                ys = [perts[i](apc.A[0])]
                ax.plot([apc.A[0]], ys, marker='o', markersize=3, c=l.get_c())
        update_ax_ylim(ax, ys)
    for ax in axs:
        ax.legend(loc='lower left', fontsize='6', labelspacing=0.05, handletextpad=0.1)

def draw_predicates_2d(apcs, labels, mesh, axs, perts=None):
    for i, apc in enumerate(apcs):
        ax = axs[apc.u_comp]
        vs = np.array([apc.A[0], apc.A[0], apc.A[1], apc.A[1]])
        vs[1][0] = vs[2][0]
        vs[3][0] = vs[0][0]
        verts = np.array([
            np.hstack([vs[i], apc.p(*vs[i])]) for i in range(vs.shape[0])])
        ax.add_collection(p3.art3d.Poly3DCollection([verts]))
        update_ax_ylim(ax, verts[:, -1])
        if perts is not None:
            elems = mesh.find_elems_between(apc.A[0], apc.A[1])
            el_values = [{
                e: perts[i](*mesh.get_elem(e, elems.dimension).chebyshev_center())
                for e in elems.elems} for i in range(len(perts))]
            for el_value in el_values:
                # logger.debug(el_value[elems.elems[0]])
                verts_pert = np.array([
                    np.hstack(
                        [elems[e], np.ones((elems[e].shape[0], 1)) * el_value[e]])
                    for e in elems.elems
                ])
                ax.add_collection(p3.art3d.Poly3DCollection(verts_pert))
                update_ax_ylim(ax, verts_pert[:, :, -1])
    # for ax in axs:
    #     ax.legend(loc='lower left', fontsize='6', labelspacing=0.05, handletextpad=0.1)

def predicates_displacement_data(apcs, labels, mesh, ax, perts=None):
    ret = {}
    for i, apc in enumerate(apcs):
        elem_set = mesh.find_elems_between(apc.A[0], apc.A[1])
        if elem_set.dimension > 1:
            logger.info("Skipping {} predicate of dimension {}".format(
                labels[i], apc.dimension))
            continue

        nodes = set([n for e in elem_set.elems
                     for n in mesh.elem_nodes(e, elem_set.dimension)])
        nodes = (sorted(list(nodes)))
        disps = np.array([apc.p(*mesh.nodes_coords[n]) for n in nodes])
        pts = np.array([mesh.nodes_coords[n] for n in nodes])

        l = ax.plot([], [], lw=1, label=labels[i])[0]
        ret[i] = {'nodes': nodes, 'disps': disps, 'pts': pts, 'l':l}

        if perts is not None:
            pts_pert = np.array([
                mesh.get_elem(e, elem_set.dimension).chebyshev_center()
                for e in elem_set.elems])
            disps_pert = np.array([perts[i](*c) for c in pts_pert])
            l_pert = ax.plot([], [], lw=1, ls='--', c=l.get_c())[0]
            ret[i].update({'pts_pert': pts_pert, 'disps_pert': disps_pert,
                           'l_pert': l_pert})


    return ret

def update_ax_ylim(ax, ys):
    m, M = np.min(ys), np.max(ys)
    m -= 0.03 * abs(M - m)
    M += 0.03 * abs(M - m)
    try:
        ax.set_zlim3d([min(ax.get_zlim3d()[0], m), max(ax.get_zlim3d()[1], M)])
    except:
        ax.set_ylim([min(ax.get_ylim()[0], m), max(ax.get_ylim()[1], M)])

def zoom_axes(ax, factor):
    lims = [ax.get_xlim(), ax.get_ylim()]
    newlims = [sum(lim)/2.0 + np.array([-.5, .5]) * (lim[1] - lim[0]) * (1 + factor)
               for lim in lims]
    ax.set_xlim(newlims[0])
    ax.set_ylim(newlims[1])


class DrawOpts(object):
    defaults = {
        'file_prefix': None,
        'plot_size_inches': (3, 2),
        'font_size': 8,
        'window_title': 'i3_7',
        'xlabel': '$x$',
        'ylabel': '$u$',
        'derivative_ylabel': '$\\frac{d}{dx} u$',
        'input_ylabel': '$U$',
        'input_xlabel': '$t$',
        'xaxis_scale': 1,
    }

    def __init__(self, dic):
        if dic is not None:
            copy = dic.copy()
        else:
            copy = {}
        for k, v in DrawOpts.defaults.items():
            setattr(self, k, copy.pop(k, v))
        if len(copy) > 0:
            raise Exception('Unrecognized options in DrawOpts: {}'.format(copy))
