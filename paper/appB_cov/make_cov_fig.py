"""Figure 11 (fig:covmc, Appendix B): the covariance of e_par between trial DMs, V (Eqs. covprof and
cov), against the Monte Carlo of run_cov_mc.py, for the one-component burst at S/N = 40.

One row of four panels (a two-column figure). (a, c): white noise of constant variance in equal
channels, at evenly spaced trial DMs; (b, d): noise whose variance rises by a factor of 5 at the
burst's peak, in unequal channels, at unevenly spaced trial DMs.
(a, b) the correlation of e_par(d_j) with e_par(d_ref), predicted (line) and Monte Carlo (points, with
       sampling errors (1 - rho^2)/sqrt(n));
(c, d) how the standard deviation of e_par changes with trial DM: the Monte Carlo ratio
       std[e_par(d_j)] / std[e_par(d_ref)] divided by the predicted ratio sqrt(V_jj / V_ref,ref).
At d_ref the prediction is exact (no shift: the variance of w n for noise of known variance), so its
Monte Carlo value differs from it only by sampling error, 1/sqrt(2 n), which all the trial DMs share
because their e_par are almost fully correlated. Dividing by the value at d_ref removes that common
fluctuation; what is left has sampling error sqrt((1 - rho^2)/n) for Gaussian e_par, with rho the
correlation with d_ref (from Var(log s_j - log s_ref) for two sample standard deviations).

Writes figures/fig_cov_mc.pdf.

usage: python make_cov_fig.py   (after run_cov_mc.py white and run_cov_mc.py selfnoise)
"""
import json
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from plotting import IBM_PALETTE, savefig, set_pub_col, set_pub_style  # noqa: E402

MC = IBM_PALETTE[3]  # Monte Carlo (orange)
CASES = [("white", "white"), ("selfnoise", "self-noise")]


def main():
    set_pub_col(1)
    set_pub_style(use_latex=False)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7,
                         "figure.constrained_layout.use": False})
    # One row: (a, b) correlation, a gap for the y-label of (c), then (c, d) standard deviation.
    fig = plt.figure(figsize=(7.1, 2.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.42, 1, 1], left=0.075, right=0.995, top=0.88,
                          bottom=0.21, wspace=0.08)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(gs[3], sharex=ax_a)
    ax_d = fig.add_subplot(gs[4], sharex=ax_a, sharey=ax_c)
    for ax in (ax_b, ax_d):
        ax.tick_params(labelleft=False)
    axs = np.array([[ax_a, ax_b], [ax_c, ax_d]])

    for col, (case, title) in enumerate(CASES):
        a = np.load(os.path.join(HERE, "results", f"cov_mc_{case}.npz"))
        num = json.loads(str(a["numbers"]))
        d, j_ref, V, V_mc = a["d"], int(a["j_ref"]), a["V"], a["V_mc"]
        n = num["n_real"]
        x = d - num["d0"]
        sd, sd_mc = np.sqrt(np.diag(V)), np.sqrt(np.diag(V_mc))
        rho = V[j_ref] / (sd * sd[j_ref])
        rho_mc = V_mc[j_ref] / (sd_mc * sd_mc[j_ref])

        # (a, b) correlation with d_ref.
        ax = axs[0, col]
        ax.plot(x, rho, color="k", lw=1.0, label="Predicted", zorder=1)
        ax.errorbar(x, rho_mc, yerr=np.clip(1 - rho**2, 0, None) / np.sqrt(n), fmt="o", ms=2.2, color=MC,
                    elinewidth=0.8, label="MC", zorder=2)
        ax.set_title(f"({'ab'[col]}) {title}", loc="left")
        if col == 0:
            ax.set_ylabel(r"Corr$[e_\parallel(d_j),\,e_\parallel(d_{\rm ref})]$")
            ax.legend(loc="lower center", frameon=False)

        # (c, d) change of the standard deviation with trial DM, Monte Carlo over predicted.
        ax = axs[1, col]
        ratio = (sd_mc / sd_mc[j_ref]) / (sd / sd[j_ref])
        err = np.sqrt(np.clip(1 - rho**2, 0, None) / n)
        ax.axhline(1, color="k", lw=1.0, zorder=1)
        ax.errorbar(x, ratio, yerr=err, fmt="o", ms=2.2, color=MC, elinewidth=0.8, zorder=2)
        ax.set_title(f"({'cd'[col]}) {title}", loc="left")
        if col == 0:
            ax.set_ylabel(r"$\dfrac{{\rm std}[e_\parallel(d_j)]\,/\,{\rm std}[e_\parallel(d_{\rm ref})]}"
                          r"{\sqrt{V_{jj}\,/\,V_{\rm ref,ref}}}$", labelpad=1)

    for ax in axs.flat:
        ax.set_xlabel(r"$d_j-d_0$ [pc cm$^{-3}$]")
    axs[1, 0].set_yticks([0.995, 1.0, 1.005])
    axs[1, 0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    axs[0, 0].set_xlim(-0.04, 0.04)
    axs[0, 0].set_xticks([-0.03, 0, 0.03])
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_cov_mc.pdf"), dpi=600, fig=fig)


if __name__ == "__main__":
    main()
