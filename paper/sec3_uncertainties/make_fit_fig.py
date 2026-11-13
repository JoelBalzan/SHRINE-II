"""Figure 6 (fig:fit, Section 3.3.3): the fit of Section 3.3 on the example burst.

Sp(d) with Sp +- sigma_S, the level Sp_max - sigma_S, the fit window (the contiguous trial DMs
above it), the fitted parabola and d_fit +- sigma_DM, all from shrine2.dm_uncertainty, and S23's
interval with sigma_S in place of their uncertainty (from the peak to where Sp + sigma_S falls
below Sp_max, interpolated between trial DMs).

Writes figures/fig_s23_fit.pdf and results/fig_s23_fit.json.

usage: python make_fit_fig.py
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
from shrine2 import dm_uncertainty  # noqa: E402


def main():
    BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE
    ex = sims.example()
    p = ex["pilot"]
    d0, d = p["d0"], ex["d"]

    res = dm_uncertainty(ex["P"], d, p["kc"], ex["dt_nu"])
    Sp, win, j_max, sig_S = res["Sp"], res["win"], res["j_max"], res["sig_S"]
    d_fit, sig_DM = res["d_fit"], res["sig_DM"]
    c0, c1, c2 = res["c"]
    info = dict(seed=sims.EXAMPLE_SEED, kc=p["kc"], d0=d0, sig_S=float(sig_S),
                window=[res["d_lo"], res["d_hi"]], n_win=res["n_win"], d_fit=float(d_fit),
                sig_DM=float(sig_DM), half_width_over_sigma=float(0.5 * (res["d_hi"] - res["d_lo"]) / sig_DM))
    print(json.dumps(info, indent=1))

    set_pub_col(1)
    set_pub_style(use_latex=False)
    plt.rcParams["figure.constrained_layout.use"] = False
    X = d - d0
    lo, hi = res["d_lo"] - d0, res["d_hi"] - d0
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    x_range = (centre - 1.45 * half, centre + 1.45 * half)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.15)

    # S23's interval with sigma_S: from the peak to where Sp + sigma_S falls below Sp_max,
    # interpolated between the trial DMs either side of each end of the window.
    j_first, j_last = np.flatnonzero(win)[[0, -1]]
    level = Sp[j_max] - sig_S
    x_low = np.interp(level, [Sp[j_first - 1], Sp[j_first]], [X[j_first - 1], X[j_first]])
    x_high = np.interp(level, [Sp[j_last + 1], Sp[j_last]], [X[j_last + 1], X[j_last]])
    ax.axvspan(x_low, x_high, color=PURP, alpha=0.10, lw=0, zorder=0, label="Fit window")
    h_Sp, = ax.plot(X, Sp, color=ORNG, lw=1.6, zorder=3)
    h_band = ax.fill_between(X, Sp - sig_S, Sp + sig_S, color=ORNG, alpha=0.18, lw=0, zorder=1)
    ax.errorbar([X[j_max]], [Sp[j_max]], xerr=[[X[j_max] - x_low], [x_high - X[j_max]]], color=MAG,
                lw=1.2, capsize=2, fmt="none", zorder=5, label=r"S23 interval with $\sigma_{\mathcal{S}}$")

    # The parabola about its vertex, Sp ~ c0 + c1 (d - d_fit) + c2 (d - d_fit)^2, over the window.
    dd = np.linspace(res["d_lo"], res["d_hi"], 200)
    ax.plot(dd - d0, c0 + c1 * (dd - d_fit) + c2 * (dd - d_fit) ** 2, color="k", lw=1.0,
            ls=(0, (4, 2)), zorder=4, label="Parabola fit")
    ax.errorbar([d_fit - d0], [c0], xerr=[sig_DM], color=PURP, fmt="o", ms=3, capsize=2, lw=1.2,
                zorder=6, label=r"$d_{\rm fit}\pm\sigma_{\rm DM}$")
    ax.axvline(0, color="k", lw=0.7, ls=(0, (1, 2)), zorder=2, label=r"$d_0$")

    ax.set_xlim(*x_range)
    shown = (X > x_range[0]) & (X < x_range[1])
    y_low, y_high = Sp[shown].min(), Sp[j_max]
    ax.set_ylim(y_low - 0.9 * (y_high - y_low), y_high + 1.12 * sig_S)
    ax.set_xlabel(r"$d-d_0$ [pc cm$^{-3}$]")
    ax.set_ylabel(r"$\mathcal{S}(d)$")
    ax.grid(True, lw=0.5, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(1, (h_band, h_Sp))
    labels.insert(1, r"Measured $\mathcal{S}\pm\sigma_{\mathcal{S}}$")
    ax.legend(handles, labels, ncol=2, fontsize=6, loc="lower center", handlelength=1.5,
              columnspacing=0.8)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_s23_fit.pdf"), dpi=600, fig=fig)
    json.dump(info, open(os.path.join(HERE, "results", "fig_s23_fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
