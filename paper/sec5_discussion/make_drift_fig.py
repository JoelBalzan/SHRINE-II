"""Figure 9 (fig:s23_drift, Section 5.1): structure maximisation for drifting sub-bursts.

(a, b) the noiseless dynamic spectrum and profile of the t_sep = 0.2 ms burst at the true DM (a) and
at d_0 for k_c set at S/N = 40 (b); (c) Sp(d) for t_sep = 0.35 ms with k_c set at S/N = 15 and 200;
(d) d_0 against t_sep at S/N = 15, 40 and 200 with error bars +-sigma_DM, and the DM that maximises
the peak of the profile. From results/drift_results.json and drift_curves.npz (run_drift.py).

Writes figures/fig_s23_drift.pdf.

usage: python make_drift_fig.py   (after run_drift.py)
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

EXAMPLE = "drift020"  # panels (a) and (b)


def main():
    BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE
    rows = json.load(open(os.path.join(HERE, "results", "drift_results.json")))
    curves = np.load(os.path.join(HERE, "results", "drift_curves.npz"))
    example = None
    for row in rows:
        if row["tag"] == EXAMPLE:
            example = row
    d0_example = example["snr"]["40"]["d0"]

    I0, freq, time = sims.load_burst(EXAMPLE)
    dt_nu = sims.channel_delays(freq, time[1] - time[0])

    set_pub_col(1)
    set_pub_style(use_latex=False)
    fig = plt.figure(figsize=(7.1, 5.6))
    fig.set_layout_engine("none")  # manual layout, so that each profile and dynamic spectrum touch
    gs = fig.add_gridspec(2, 2, height_ratios=[1.32, 1.15], hspace=0.36, wspace=0.26, left=0.09,
                          right=0.99, top=0.95, bottom=0.09)
    vmax = np.percentile(I0, 99.8)

    # (a, b) at the true DM and at d_0.
    panels = [(0.0, r"(a) $\mathrm{DM}_\mathrm{true}$"),
              (d0_example, rf"(b) $d_0=\mathrm{{DM}}_\mathrm{{true}}+{d0_example:.3f}$ pc cm$^{{-3}}$")]
    for i, (dm, title) in enumerate(panels):
        D = sims.dedisperse_spectrum(I0, dm, dt_nu)
        prof = D.sum(axis=0)
        sub = gs[0, i].subgridspec(2, 1, height_ratios=[0.32, 1.0], hspace=0)
        ax_prof = fig.add_subplot(sub[0])
        ax_ds = fig.add_subplot(sub[1], sharex=ax_prof)
        ax_prof.plot(time, prof / prof.max(), color="k", lw=0.9)
        ax_prof.set_ylim(-0.05, 1.15)
        ax_prof.set_title(title, loc="left")
        ax_prof.tick_params(labelbottom=False, labelleft=False, left=False, right=False)
        ax_prof.grid(True, lw=0.5, alpha=0.5)
        ax_ds.imshow(D, aspect="auto", origin="upper", extent=[time[0], time[-1], freq[-1], freq[0]],
                     cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
        ax_ds.set_xlim(-0.5, 1.5)
        ax_ds.set_xlabel("$t$ [ms]")
        if i == 0:
            ax_ds.set_ylabel("Frequency [MHz]")
        else:
            ax_ds.tick_params(labelleft=False)

    # (c) Sp(d) for t_sep = 0.35 ms, which has two maxima.
    ax = fig.add_subplot(gs[1, 0])
    d = curves["d"]
    for Sp, colour, label in [(curves["Sp_low"], ORNG, rf"$k_c={int(curves['kc_low'])}$ (S/N = 15)"),
                              (curves["Sp_high"], PURP, rf"$k_c={int(curves['kc_high'])}$ (S/N = 200)")]:
        ax.plot(d, Sp / Sp.max(), color=colour, lw=1.4, label=label)
    for row in rows:
        if row["tag"] == "drift035":
            dm_peak_035 = row["dm_peak"]
    ax.axvline(0, color="0.3", lw=0.8, ls=":")
    ax.axvline(dm_peak_035, color="k", lw=0.8, ls="--")
    ax.set_xlim(d[0], d[-1])
    ax.set_xlabel(r"$d-\mathrm{DM}_\mathrm{true}$ [pc cm$^{-3}$]")
    ax.set_ylabel(r"$\mathcal{S}(d)/\max\mathcal{S}$")
    ax.set_title(r"(c) Structure curve, $t_{\rm sep}=0.35$ ms", loc="left")
    ax.legend(loc="lower center", fontsize=8)
    ax.grid(True, lw=0.5, alpha=0.5)

    # (d) d_0 against t_sep.
    ax = fig.add_subplot(gs[1, 1])
    t_sep = []
    dm_peak = []
    for row in rows:
        t_sep.append(row["t_sep"])
        dm_peak.append(row["dm_peak"])
    t_sep = np.array(t_sep)
    ax.plot(t_sep, dm_peak, color="k", ls="none", marker="x", ms=5, mew=1.0, label="Peak-maximising DM")
    for i, (snr, colour, marker) in enumerate([("15", ORNG, "o"), ("40", BLUE, "s"), ("200", PURP, "D")]):
        d0 = []
        sig = []
        for row in rows:
            d0.append(row["snr"][snr]["d0"])
            sig.append(row["snr"][snr]["sig_DM"])
        ax.errorbar(t_sep + 0.006 * (i - 1), d0, yerr=sig, color=colour, marker=marker, ms=4, lw=1.0,
                    capsize=0, label=f"S/N = {snr}")
    ax.axhline(0, color="0.3", lw=0.8, ls=":")
    ax.set_xlabel(r"Sub-burst separation $t_{\rm sep}$ [ms]")
    ax.set_ylabel(r"$d_0-\mathrm{DM}_\mathrm{true}$ [pc cm$^{-3}$]")
    ax.set_title("(d) Structure-maximising DM", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=7)
    ax.grid(True, lw=0.5, alpha=0.5)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_s23_drift.pdf"), dpi=600, fig=fig)


if __name__ == "__main__":
    main()
