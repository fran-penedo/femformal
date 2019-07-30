import imp
import traceback

import numpy as np
import matplotlib.pyplot as plt


prefix = "examples/heat_mix/results/hm_synth_simple2_N{}_results.py"
N = list(range(10, 101, 10))


def load_module(name, f):
    try:
        return imp.load_source(name, f)
    except Exception:
        print("Couldn't load module {}".format(f))
        traceback.print_exc()
        return None


robustness = []
times = []
for n in N:
    f = prefix.format(str(n))
    name = f.replace("/", ".")
    m = load_module(name, f)
    if m is not None:
        robustness.append(m.robustness)
        times.append(m.time)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.plot(N, robustness, color="r", ms=5, ls="none", marker="o")
ax1.set_xlabel("Mesh Number of Elements")
ax1.set_ylabel("Robustness Bound")

ax2.bar(N, times)
ax2.set_ylabel("Computing Time (s)")

fig.set_size_inches((3, 2))
for ax in fig.get_axes():
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
        item.set_fontsize(8)
fig.tight_layout()

# plt.show()
fig.savefig("temp_plots/hm_simple2_rob_vs_time.png", bbox_inches="tight", pad_inches=0)
