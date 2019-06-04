import os

import numpy as np
import matplotlib.pyplot as plt

DIR = "examples/heat_mix/results"
FILE_PATTERN = "hm_verif_simple2_{}_results_N{}.py"

ns = range(10, 101, 10)
corrections = range(4)

if __name__ == "__main__":
    table = []
    for n in ns:
        line = []
        for c in corrections:
            fn = os.path.join(DIR, FILE_PATTERN.format(c, n))
            with open(fn, 'r') as f:
                exec(f.read())
                line.append('{:.3f}'.format(robustness))
        table.append(line)

    table = np.array(table)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    handles = ax1.plot(ns, table, ms=5, ls='-', marker='o')
    ax1.set_xlabel('Mesh Number of Elements')
    ax1.set_ylabel('Robustness Bound')
    ax1.grid(which='major', axis='y')

    labels = ['$r(\phi_{FEM}^{0, 0, 0})$', '$r(\phi_{FEM}^{\delta, 0, 0})$',
              '$r(\phi_{FEM}^{\delta, \eta, 0})$',
              '$r(\phi_{FEM}^{\delta, \eta, \\nu})$']
    fig.legend(handles, labels, ncol=2, fontsize=10, columnspacing=0.5,
               labelspacing=0.1, loc='center left', bbox_to_anchor=(0.15, 1.15), numpoints=1)

    fig.set_size_inches((3,2))
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label,
                    ax.yaxis.get_offset_text(), ax.xaxis.get_offset_text()] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
    # fig.tight_layout()

    # plt.show()
    fig.savefig("temp_plots/hm_simple2_verif.png", bbox_inches='tight', pad_inches=0)
