"""Figure 3 (fig:s23est, Section 2.3): S23's uncertainties for the example burst.

The measured Sp(d) of the example burst, S23's uncertainty of Sp at the highest trial DM d_max,
||D1 S Delta_x||, and S23's DM interval about d_max (Eq. s23int), against the 1-sigma scatter
across the Monte Carlo realisations of Section 4 (one-component burst, S/N = 40,
sec4_validation/results/mc_single_snr40.npz): of d_max, and of Sp at the trial DM nearest d_0.
Both scatters are drawn about this realisation's (d_max, Sp_max). Also gives the numbers of
Section 2.3: S23's uncertainty of Sp and their DM interval relative to these scatters.

Writes figures/fig_s23_s23est.pdf and results/fig_s23_s23est.json.

usage: python make_s23est_fig.py   (after sec4_validation/run_mc.py single 40)
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from plotting import IBM_PALETTE, savefig, set_pub_col, set_pub_style  # noqa: E402
from shrine2.structure import forward_difference_matrix, smooth, structure_parameter  # noqa: E402


def main():
    BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE
    ex = sims.example()
    p = ex["pilot"]
    kc, d0, d, P = p["kc"], p["d0"], ex["d"], ex["P"]

    Sp = structure_parameter(P, kc)
    j_max = int(np.argmax(Sp))

    # S23's uncertainty of Sp at d_max, ||D1 S Delta_x||, and their DM interval.
    D1 = forward_difference_matrix(P.shape[1])
    s23_uncertainty = np.linalg.norm(D1 @ smooth(P[j_max] - smooth(P[j_max], kc), kc))
    s23_lo, s23_hi, at_edge = sims.s23_interval(P, d, kc)
    assert not at_edge, "S23's interval reaches the end of the trial DMs"

    # The Monte Carlo scatter of d_max and of Sp near d_0 (same burst, S/N, k_c and DM step).
    mc = np.load(sims.results_path("mc_single_snr40.npz"))
    meta = json.loads(str(mc["meta"]))
    assert meta["kc"] == kc and np.isclose(meta["step"], d[1] - d[0])
    sig_mc_dmax = float(np.std(mc["d_max"]))
    sig_mc_Sp = float(np.std(mc["Sp_d0"]))

    info = dict(seed=sims.EXAMPLE_SEED, kc=kc, d0=d0, d_max=float(d[j_max]), Sp_max=float(Sp[j_max]),
                s23_uncertainty=float(s23_uncertainty), s23_interval=[float(s23_lo), float(s23_hi)],
                s23_halfwidth=float(0.5 * (s23_hi - s23_lo)), sigma_mc_dmax=sig_mc_dmax,
                sigma_mc_Sp=sig_mc_Sp, n_mc=int(mc["d_max"].size),
                ratio_dm=float(0.5 * (s23_hi - s23_lo) / sig_mc_dmax),
                ratio_Sp=float(s23_uncertainty / sig_mc_Sp))
    print(json.dumps(info, indent=1))

    set_pub_col(1)
    set_pub_style(use_latex=False)
    plt.rcParams["figure.constrained_layout.use"] = False
    X = d - d0
    x_max, y_max = d[j_max] - d0, Sp[j_max]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.16)
    ax.plot(X, Sp, color=ORNG, lw=1.6, zorder=3, label=r"Measured $\mathcal{S}$")
    ax.errorbar([x_max], [y_max], yerr=[s23_uncertainty], color=MAG, lw=1.2, capsize=2, zorder=5,
                fmt="none",
                label=r"S23: DM interval, $\pm\|\mathbf{D}_1\mathbf{S}\boldsymbol{\Delta}_{\mathbf{x}}\|$")
    ax.annotate("", xy=(s23_lo - d0, y_max), xytext=(s23_hi - d0, y_max),
                arrowprops=dict(arrowstyle="|-|", color=MAG, lw=1.2, shrinkA=0, shrinkB=0,
                                mutation_scale=3), zorder=5)
    ax.errorbar([x_max], [y_max], xerr=[sig_mc_dmax], yerr=[sig_mc_Sp], color="0.25", lw=1.6,
                capsize=2, zorder=6, fmt="none", label=r"Scatter: $\pm\sigma_{\rm MC}$")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(y_max - 2.9 * s23_uncertainty, y_max + 1.25 * s23_uncertainty)
    ax.set_xlabel(r"$d-d_0$ [pc cm$^{-3}$]")
    ax.set_ylabel(r"$\mathcal{S}(d)$")
    ax.grid(True, lw=0.5, alpha=0.5)
    ax.legend(fontsize=6, loc="lower center", handlelength=1.5, columnspacing=0.8)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_s23_s23est.pdf"), dpi=600, fig=fig)
    json.dump(info, open(os.path.join(HERE, "results", "fig_s23_s23est.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
