"""Figure 10 (fig:kc, Appendix A): S23's k_c search and its extension, on one realisation of the
two-component burst at S/N = 40 for which S23's procedure stops at the dip between the burst's two
structure scales.

M(k), the largest |DCT coefficient| of mode k over the coarse trial DMs (Eq. Mk), its six-mode running
mean Mbar(k) (Eq. runmean), the noise floor M_n and the level M_n + 4 sigma_Mbar, S23's k_c and the
extended search's k_c, all from shrine2.structure.kc_search.
(a) the whole spectrum, log-log, with the modes used for M_n and sigma_Mbar and those used for
    sigma_t^2 (k > k_min = max(N/4, 2 k_c)); (b) the first 75 modes.

The realisation is number REALISATION (0-based) of the Section 4 Monte Carlo of that case (the
same seed and pilot), so its k_c values are those of run_mc_kc.py.

Writes figures/fig_kc_search.pdf and results/fig_kc_search.json.

usage: python make_kc_fig.py [realisation]
"""
import json
import os
import sys

import matplotlib
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from plotting import IBM_PALETTE, savefig, set_pub_col, set_pub_style  # noqa: E402
from shrine2.structure import kc_search  # noqa: E402

TAG, SNR = "double", 40
REALISATION = 2
K_ZOOM = 75  # modes shown in (b)


def y_ticks(ax, ticks):
    """Plain-number ticks on a log axis."""
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE
    realisation = REALISATION
    if len(sys.argv) > 1:
        realisation = int(sys.argv[1])

    I0, freq, time = sims.load_burst(TAG)
    dt_nu = sims.channel_delays(freq, time[1] - time[0])
    sc = sims.channel_noise_sigma(I0, SNR)
    _, seed = sims.CASES[(TAG, SNR)]
    rng = np.random.default_rng(seed)
    sims.pilot(I0, dt_nu, sc, rng)  # the pilot's draws, so that the realisations match run_mc.py
    for _ in range(realisation):
        rng.normal(0, sc, I0.shape)
    noise = rng.normal(0, sc, I0.shape)

    coarse = sims.coarse_grid(dt_nu)
    search = kc_search(sims.dedisperse_grid(I0 + noise, *coarse, dt_nu))
    noiseless = kc_search(sims.dedisperse_grid(I0, *coarse, dt_nu))
    M, Mbar, Mn, rise_level = search["M"], search["Mbar"].copy(), search["Mn"], search["rise_level"]
    Mbar[:5] = np.nan  # incomplete six-mode windows
    kc_s23, kc_ext = search["kc_s23"], search["kc_ext"]
    N = M.size
    k_min = max(N // 4, 2 * kc_ext)
    info = dict(tag=TAG, snr=SNR, seed=seed, realisation=realisation, kc_s23=int(kc_s23),
                kc_ext=int(kc_ext), last_rise=search["last_rise"], Mn=float(Mn),
                rise_level=float(rise_level), k_min=int(k_min))
    print(info)

    set_pub_col(1)
    set_pub_style(use_latex=False)
    plt.rcParams["figure.constrained_layout.use"] = False
    k = np.arange(N)
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.6), gridspec_kw=dict(height_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.09, hspace=0.38)

    # (a) the whole spectrum.
    ax = axs[0]
    ax.axvspan(k_min, N, color=YELL, alpha=0.22, lw=0, zorder=0,
               label=r"$\sigma_t^2$ modes, $k\geq k_{\min}$")
    ax.axvspan(N // 2, N, color="0.6", alpha=0.35, lw=0, zorder=0,
               label=r"$M_{\rm n}$, $\sigma_{\bar M}$ modes, $k\geq N/2$")
    ax.plot(k[1:], M[1:], color="0.55", lw=0.5, label=r"$M(k)$")
    ax.plot(k[1:], noiseless["M"][1:], color="k", lw=0.9, label="Noiseless")
    ax.axhline(Mn, color=BLUE, lw=0.9, ls="--", label=r"$M_{\rm n}$")
    ax.axvline(kc_s23, color=MAG, lw=1.0, ls=":")
    ax.axvline(kc_ext, color=PURP, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, N)
    ax.set_ylim(np.percentile(M[N // 2:], 1) * 0.6, M.max() * 2)
    ax.set_xlabel(r"DCT mode $k$")
    ax.set_ylabel(r"$M(k)$")
    ax.set_title("(a)", loc="left")
    y_ticks(ax, [20, 50, 100, 200])
    ax.legend(fontsize=6, loc="upper right", handlelength=1.6)
    ax.grid(True, lw=0.5, alpha=0.5)

    # (b) the first K_ZOOM modes.
    ax = axs[1]
    ax.axvspan(5, 2 * kc_s23, color=PURP, alpha=0.07, lw=0, zorder=0,
               label=r"Extended search, $k<2k_c$")
    ax.plot(k[1:K_ZOOM], M[1:K_ZOOM], color="0.55", lw=0.6, marker="o", ms=1.6, label=r"$M(k)$")
    ax.plot(k[1:K_ZOOM], noiseless["M"][1:K_ZOOM], color="k", lw=0.9, label="Noiseless")
    ax.plot(k[:K_ZOOM], Mbar[:K_ZOOM], color=ORNG, lw=1.4, label=r"$\bar M(k)$ (six-mode mean)")
    ax.axhline(Mn, color=BLUE, lw=0.9, ls="--", label=r"$M_{\rm n}$")
    ax.axhline(rise_level, color=BLUE, lw=0.9, ls=":", label=r"$M_{\rm n}+4\sigma_{\bar M}$")
    ax.axvline(kc_s23, color=MAG, lw=1.0, ls=":", label=fr"S23 $k_c={kc_s23}$")
    ax.axvline(kc_ext, color=PURP, lw=1.0, label=fr"Extended $k_c={kc_ext}$")
    ax.set_xlim(0, K_ZOOM)
    ax.set_yscale("log")
    ax.set_ylim(Mn * 0.45, np.nanmax(noiseless["M"][5:K_ZOOM]) * 3.2)
    y_ticks(ax, [20, 50, 100, 200])
    ax.set_xlabel(r"DCT mode $k$")
    ax.set_ylabel(r"$M(k)$")
    ax.set_title("(b)", loc="left")
    ax.legend(fontsize=5.8, loc="upper right", ncol=2, handlelength=1.6, columnspacing=0.8)
    ax.grid(True, lw=0.5, alpha=0.5)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_kc_search.pdf"), dpi=600, fig=fig)
    json.dump(info, open(os.path.join(HERE, "results", "fig_kc_search.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
