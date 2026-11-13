"""Figure 1 (fig:burst, Section 2.1): the example burst at the true DM.

(a) the dynamic spectrum (channels averaged by 8 and samples by 2, for display);
(b) the frequency-summed profile x, the noiseless burst s and the smoothed profile S x;
(c) the structure vector D1 S x, its noiseless part a = D1 S s and S23's smoothed residual
    D1 S Delta_x (Delta_x = x - S x).
The norms of the vectors in (c) are Sp, ||a|| and S23's uncertainty of Sp.

Writes figures/fig_s23_burst.pdf and results/fig_s23_burst.json.

usage: python make_burst_fig.py
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

N_AVG, T_BIN, T_WINDOW = 8, 2, (-0.4, 1.4)  # display only: channels and samples averaged, ms shown


def main():
    BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE
    ex = sims.example()
    I0, freq, time, sc = ex["I0"], ex["freq"], ex["time"], ex["sc"]
    kc = ex["pilot"]["kc"]
    n_chan = freq.size

    # At the true DM (d = 0) dedispersion leaves the dynamic spectrum as it is.
    D = I0 + ex["noise"]
    s = I0.sum(axis=0)
    x = D.sum(axis=0)

    # The vectors of panel (c).
    D1 = forward_difference_matrix(x.size)
    Sx = smooth(x, kc)
    Ss = smooth(s, kc)
    D1Sx = D1 @ Sx
    a = D1 @ Ss
    D1S_residual = D1 @ smooth(x - Sx, kc)
    assert np.isclose(np.linalg.norm(D1Sx), structure_parameter(x, kc))
    info = dict(seed=sims.EXAMPLE_SEED, kc=kc, Sp=float(np.linalg.norm(D1Sx)),
                norm_a=float(np.linalg.norm(a)), s23_uncertainty=float(np.linalg.norm(D1S_residual)),
                sigma_profile=float(np.sqrt(n_chan) * sc))
    print(info)

    # Display: average channels by N_AVG and samples by T_BIN over T_WINDOW.
    tw = np.flatnonzero((time >= T_WINDOW[0]) & (time <= T_WINDOW[1]))
    tw = tw[: (tw.size // T_BIN) * T_BIN]
    nf = (n_chan // N_AVG) * N_AVG
    W = D[:nf, tw].reshape(-1, N_AVG, tw.size).mean(axis=1)
    W = W.reshape(nf // N_AVG, -1, T_BIN).mean(axis=2)
    f_binned = freq[:nf].reshape(-1, N_AVG).mean(axis=1)
    if f_binned[0] > f_binned[-1]:
        W = W[::-1]
        f_binned = f_binned[::-1]
    t_binned = time[tw].reshape(-1, T_BIN).mean(axis=1)
    dt = time[1] - time[0]
    t_mid = 0.5 * (time[1:] + time[:-1])  # D1 v sits between samples
    sig = np.sqrt(n_chan) * sc  # profile noise per sample: profiles are plotted in units of it

    set_pub_col(1)
    set_pub_style(use_latex=False)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axs = plt.subplots(3, 1, figsize=(3.4, 4.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.985, bottom=0.09, hspace=0.07)

    ax = axs[0]
    vmin, vmax = np.percentile(W, [5, 99])
    ax.imshow(W, aspect="auto", origin="lower", cmap="plasma", vmin=vmin, vmax=vmax,
              interpolation="nearest",
              extent=[t_binned[0] - T_BIN * dt / 2, t_binned[-1] + T_BIN * dt / 2,
                      f_binned[0], f_binned[-1]])
    ax.tick_params(axis="both", which="both", direction="in", color="white", labelcolor="black")
    ax.set_ylabel("Frequency [MHz]")
    ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, ha="left", va="top", color="white",
            fontsize=8)

    ax = axs[1]
    ax.plot(time, x / sig, color="0.6", lw=0.6, zorder=2, label=r"$\mathbf{x}$")
    ax.plot(time, s / sig, color="k", lw=1.0, zorder=3, label=r"$\mathbf{s}$")
    ax.plot(time, Sx / sig, color=ORNG, lw=1.2, zorder=4, label=r"$\mathbf{S}\mathbf{x}$")
    ax.set_ylabel("S/N")
    ax.legend(fontsize=6, loc="upper right", handlelength=1.5)
    ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=8)

    ax = axs[2]
    ax.axhline(0, color="0.3", lw=0.5, zorder=1)
    ax.plot(t_mid, D1Sx / sig, color=ORNG, lw=1.2, zorder=3,
            label=r"$\mathbf{D}_1\mathbf{S}\mathbf{x}$")
    ax.plot(t_mid, a / sig, color="k", lw=1.0, ls=(0, (4, 2)), zorder=4,
            label=r"$\mathbf{a}=\mathbf{D}_1\mathbf{S}\mathbf{s}$")
    ax.plot(t_mid, D1S_residual / sig, color=MAG, lw=1.0, zorder=5,
            label=r"$\mathbf{D}_1\mathbf{S}\boldsymbol{\Delta}_{\mathbf{x}}$")
    ax.set_ylabel(r"$\Delta(\mathrm{S/N})$")
    ax.legend(fontsize=6, loc="upper right", handlelength=1.5)
    ax.text(0.03, 0.95, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=8)

    for ax in axs[1:]:
        ax.grid(True, lw=0.5, alpha=0.5)
    axs[2].set_xlim(*T_WINDOW)
    axs[2].set_xlabel("Time [ms]")
    fig.align_ylabels(axs)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_s23_burst.pdf"), dpi=600, fig=fig)
    json.dump(info, open(os.path.join(HERE, "results", "fig_s23_burst.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
