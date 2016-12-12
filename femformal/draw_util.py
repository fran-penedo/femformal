import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_figcounter = 0
_holds = []

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


def draw_pde_trajectory(ds, xs, ts, prefix=None, hold=False):
    global _figcounter
    global _holds

    print ds.shape, xs.shape
    d_min, d_max = np.amin(ds), np.amax(ds)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(d_min, d_max)
    ax.set_xlabel('x')
    ax.set_ylabel('u')

    l, = ax.plot([], [], 'b-')
    time_text = ax.text(.02, .95, '', transform=ax.transAxes)

    def update_line(i):
        l.set_data(xs, ds[i])
        time_text.set_text('t = {}'.format(ts[i]))
        return l, time_text

    frames = min(len(ts), len(ds))
    line_ani = animation.FuncAnimation(
        fig, update_line, frames=frames, interval=5000/frames, blit=True)
    _holds.append(line_ani)

    if not hold:
        if prefix is not None:
            line_ani.save(prefix + str(_figcounter) + '.mp4')
            plt.close(fig)
            plt.show()
        else:
            plt.show()
        _holds = []

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




