"""Figure 7 (fig:s23_val, Section 4): validation on the simulated bursts.

(a, b) the distribution of d_fit - d_0 over the realisations at S/N = 40 for one (a) and two (b)
components (insets: the noiseless frequency-summed profiles), with the median S23 interval, the median
sigma_S crossing range and a Gaussian of width the median quoted sigma_DM;
(c, d) the median quoted uncertainty of the DM (c) and of Sp (d) divided by the Monte Carlo scatter,
against S/N, for this work, the sigma_S crossing range (c only) and S23. Arrows mark S23 intervals that
reached the end of their trial DMs in more than 5% of realisations (lower limits).
From the results of run_mc.py, through summarise.case_numbers.

Writes figures/fig_s23_validation.pdf.

usage: python make_validation_fig.py   (after run_mc.py for every case)
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
sys.path.insert(0, HERE)
import sims  # noqa: E402
from plotting import IBM_PALETTE, pub_grid_figsize, savefig, set_pub_col, set_pub_style  # noqa: E402
from summarise import case_numbers  # noqa: E402

NEW, OLD = IBM_PALETTE[1], IBM_PALETTE[2]  # this work (purple), S23 (magenta)
BAR, PROF = IBM_PALETTE[3], "k"  # Monte Carlo histogram (orange), profiles (black)
CROSS = IBM_PALETTE[0]  # sigma_S crossing range (blue)
LABELS = dict(new="This work", cross=r"$\sigma_{\mathcal{S}}$ crossings", s23="S23")


def hist_panel(ax, tag, title, legend=True):
    """Panel (a) or (b): d_fit - d_0 at S/N = 40."""
    mc = np.load(sims.results_path(f"mc_{tag}_snr40.npz"))
    meta = json.loads(str(mc["meta"]))
    ok = np.isfinite(mc["d_fit"]) & np.isfinite(mc["sig_DM"])
    x = mc["d_fit"][ok] - meta["d0"]
    sig = np.median(mc["sig_DM"][ok])
    s23 = np.nanmedian(0.5 * (mc["s23_hi"] - mc["s23_lo"]))
    h_cross = np.nanmedian(mc["h_cross"])

    ax.axvspan(-s23, s23, color=OLD, alpha=0.08, lw=0, zorder=0)
    for v in (-s23, s23):
        ax.axvline(v, color=OLD, lw=1.0, ls="--", zorder=1)
    for v in (-h_cross, h_cross):
        ax.axvline(v, color=CROSS, lw=1.2, ls=":", zorder=1)
    bins = np.linspace(-4 * sig, 4 * sig, 25)
    ax.hist(x, bins=bins, density=True, color=BAR, alpha=0.45, edgecolor="white", lw=0.4, zorder=2,
            label="MC")
    g = np.linspace(-4 * sig, 4 * sig, 400)
    ax.plot(g, np.exp(-0.5 * (g / sig) ** 2) / (np.sqrt(2 * np.pi) * sig), color=NEW, lw=1.5, zorder=3,
            label=r"$\sigma_{\rm DM}$")
    ax.plot([], [], color=CROSS, ls=":", lw=1.2, label=r"$\sigma_{\mathcal{S}}$ crossings")
    ax.plot([], [], color=OLD, ls="--", lw=1.0, label="S23 interval")
    ax.set_xlim(-1.12 * s23, 1.12 * s23)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$d_{\rm fit}-d_0$ [pc cm$^{-3}$]")
    ax.set_ylabel("Probability density")
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.set_title(title, loc="left")
    ax.grid(True, lw=0.5, alpha=0.5)
    if legend:
        ax.legend(loc="upper left")

    # Inset: the noiseless frequency-summed profile over 3 ms.
    I0, freq, time = sims.load_burst(tag)
    inset = ax.inset_axes([0.64, 0.56, 0.33, 0.40])
    shown = (time > -0.5) & (time < 2.5)
    prof = I0.sum(axis=0)
    inset.plot(time[shown], prof[shown] / prof.max(), color=PROF, lw=0.9)
    inset.set_xlabel("$t$ [ms]", fontsize=8, labelpad=1)
    inset.tick_params(labelsize=7, which="both", left=False, right=False, labelleft=False, top=False)
    inset.minorticks_off()
    return dict(sig_DM=sig, s23=s23, h_cross=h_cross, sig_mc=x.std())


def ratio_panel(ax, rows_by_tag, quantity, title):
    """Panel (c) (quantity "dm") or (d) ("sp")."""
    if quantity == "dm":
        series = [("new", "q", NEW, "-"), ("cross", "cross_over_mc", CROSS, ":"), ("s23", "s23_over_mc", OLD, "--")]
        error_key = "q_err"
    else:
        series = [("new", "sp_new", NEW, "-"), ("s23", "sp_s23", OLD, "--")]
        error_key = "sp_err"
    for tag, marker, short in [("single", "o", "1C"), ("double", "s", "2C")]:
        rows = rows_by_tag[tag]
        snr = []
        errors = []
        for row in rows:
            snr.append(row["snr"])
            errors.append(row[error_key])
        x = np.array(snr) + (0.0 if tag == "single" else 3.0)  # 2C shifted right for clarity
        for name, key, colour, ls in series:
            y = []
            for row in rows:
                y.append(row[key])
            y = np.array(y)
            face = colour if tag == "single" else "white"  # 1C filled, 2C open
            ax.errorbar(x, y, yerr=np.array(errors) * y, color=colour, ls=ls, marker=marker, ms=4.5, mfc=face,
                        mec=colour, mew=1.0, capsize=0, label=LABELS[name] + ", " + short)
            if quantity == "dm" and name == "s23":
                for xi, yi, row in zip(x, y, rows):
                    if row["s23_at_edge"] > 0.05:  # S23's interval reached the end of the trial DMs
                        ax.annotate("", xy=(xi, yi * 1.5), xytext=(xi, yi * 1.08),
                                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.9, mutation_scale=7))
    ax.axhline(1, color="0.3", lw=0.8, zorder=1)
    ax.set_xlim(0, 215)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 30)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.grid(True, which="major", lw=0.5, alpha=0.5)
    ax.set_xlabel("S/N")
    if quantity == "dm":
        ax.set_ylabel(r"$\sigma_{\rm DM}/\sigma_{\rm MC}$")
    else:
        ax.set_ylabel(r"$\sigma_{\mathcal{S}}/\sigma_{\rm MC}$")
    ax.set_title(title, loc="left")


def main():
    rows_by_tag = {}
    for tag in ["single", "double"]:
        rows_by_tag[tag] = []
        for snr in sims.SNRS:
            row, _ = case_numbers(tag, snr)
            rows_by_tag[tag].append(row)

    set_pub_col(1)
    set_pub_style(use_latex=False)
    fig, axs = plt.subplots(2, 2, figsize=pub_grid_figsize(n_rows=2, ncol=1, row_height=3.0))
    a = hist_panel(axs[0, 0], "single", "(a) One component, S/N = 40")
    b = hist_panel(axs[0, 1], "double", "(b) Two components, S/N = 40", legend=False)
    ratio_panel(axs[1, 0], rows_by_tag, "dm", "(c) DM uncertainty")
    ratio_panel(axs[1, 1], rows_by_tag, "sp", "(d) Structure-parameter uncertainty")
    axs[1, 0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.2), ncol=2, columnspacing=0.8)
    print("S/N 40, pc cm^-3: one component", a, "two components", b)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_s23_validation.pdf"), dpi=600, fig=fig)


if __name__ == "__main__":
    main()
