import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import matplotlib.colors as colors
import matplotlib.cm as cmx
from bisect import bisect_left, bisect_right

_figcounter = 0
_holds = []

def draw_linear(ys, xs, ylabel='y', xlabel='x'):
    matplotlib.rcParams.update({'font.size': 8})
    matplotlib.rcParams.update({'figure.autolayout': True})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-100, 6e3)
    ax.plot(xs, ys, '-')


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

def draw_pde_trajectories(dss, xss, tss, pwc=False, ylabel="u", xlabel="x"):
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
                ls[j].set_color(scalarmap.to_rgba(next_t))
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
                ls[j].set_color(scalarmap.to_rgba(next_t))
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
    if scalarmap is None:
        cmap = plt.get_cmap('autumn')
        cnorm = colors.Normalize(ts[0], ts[-1])
        scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        scalarmap.set_array(ts)

    if hlines:
        l = ax.add_collection(LineCollection([]))
        # _der_lines(ax, xs)
        def update_line(i, l=l):
            l.set_segments([np.array([[xs[j], ds[i][j]], [xs[j+1], ds[i][j]]])
                            for j in range(len(xs) - 1)])
            # l[j].set_data(xs[j:j+2], [ds[i][j], ds[i][j]])
            l.set_color(scalarmap.to_rgba(ts[i]))
            time_text.set_text('t = {}'.format(ts[i]))
            return l, time_text


    else:
        l, = ax.plot([], [], 'b-')

        def update_line(i, l=l):
            l.set_data(xs, ds[i])
            l.set_color(scalarmap.to_rgba(ts[i]))
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
        interval=5000/frames, blit=True)

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
            interval=5000/frames, blit=True)
        fig.__line_ani = line_ani

    slider.on_changed(onChanged)
    slider.drawon = False
    # lines(0)
    fig.__slider = slider
    fig.__line_ani = line_ani

    return line_ani


def draw_pde_trajectory(ds, xs, ts, animate=True, prefix=None, hold=False,
                        ylabel='Temperature', xlabel='x', allonly=None):
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
                            scalarmap=scalarmap)
    line_bl = set_traj_line(ax_bl, ds, xs, ts, xlabel=xlabel, ylabel=ylabel,
                            scalarmap=scalarmap)
    dds = np.true_divide(np.diff(ds), np.diff(xs))
    line_tr = set_traj_line(ax_tr, dds, xs, ts, hlines=True, xlabel=xlabel,
                            ylabel="$\\frac{d}{d x}$" + ylabel, scalarmap=scalarmap)
    line_br = set_traj_line(ax_br, dds, xs, ts, hlines=True, xlabel=xlabel,
                            ylabel="$\\frac{d}{d x}$" + ylabel, scalarmap=scalarmap)

    savefun = None
    # if animate:
    #     frames = min(len(ts), len(ds))
    #     line_ani = animation.FuncAnimation(
    #         fig, _combine_lines([line_tl, line_tr]), frames=frames,
    #         interval=5000/frames, blit=True)
    #     _holds.append(line_ani)
    #     if prefix:
    #         savefun = lambda: line_ani.save(prefix + str(_figcounter) + '.mp4')
    # else:
    #     fig.subplots_adjust(bottom=0.1)
    #     axslider = plt.axes([0.0, 0.0, 1, .03])
    #     slider = Slider(axslider, 'Time', ts[0], ts[-1], valinit=ts[0])
    #     lines = _combine_lines([line_tl, line_tr])
    #     slider.on_changed(lambda val: lines(bisect_left(ts, val)))
    #     lines(0)
    #     fig.__slider = slider

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
    fig.tight_layout()

    _render(fig, savefun, hold)


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

