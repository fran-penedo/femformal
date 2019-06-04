import numpy as np
import matplotlib.pyplot as plt

tols = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
rs = [0.474270751663, 0.29109354188, 0.107916445318, -0.0752607125047, -0.258437867235, -0.44161502227]
ts = [30.3271350861, 27.9897711277, 28.3531999588, 34.7827649117, 30.8727140427, 32.5704009533]

if __name__ == "__main__":
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(np.array(tols) * 100, rs, color='r', ms=5, ls='', marker='o')
    ax1.set_xlabel('Input Tolerance (%)')
    ax1.set_ylabel('Robustness Bound')
    ax1.grid(which='major', axis='y')

    fig.set_size_inches((3,2))
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label,
                    ax.yaxis.get_offset_text(), ax.xaxis.get_offset_text()] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
    fig.tight_layout()

    # plt.show()
    fig.savefig("temp_plots/hm_verif_set_simple2.png", bbox_inches='tight', pad_inches=0)
