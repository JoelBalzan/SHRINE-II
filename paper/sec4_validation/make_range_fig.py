"""Figure 8 (fig:s23_range, Section 4): the example burst dedispersed to the edges of each interval.

The example burst (one-component burst, S/N = 40, seed sims.EXAMPLE_SEED) dedispersed to the lower
edge, optimum and upper edge of
    top row:    S23's DM interval about the highest trial DM d_max (Eq. s23int; sims.s23_interval);
    bottom row: d_fit -+ sigma_DM (Section 3.3; shrine2.dm_uncertainty),
in the style of FRBop's dmop --plot-range: waterfalls with shared 5-99 percentile colour limits and a
white line at the leading edge of the pulse in the top quarter of the band (where the 5-sample
running mean of that band's profile first reaches 10% of its peak, at the optimum), moved in the
other panels by the change in that band's dispersion delay. For display only, channels are averaged
by N_AVG and samples by T_BIN.

Also gives the delay between the band edges that each offset from the optimum introduces (Section 4:
S23's edges shift the bottom of the band by more than the burst's envelope, d_fit +- sigma_DM by about
two samples).

Writes figures/fig_s23_range.pdf and results/fig_s23_range.json.

usage: python make_range_fig.py
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
from plotting import savefig, set_pub_col, set_pub_style  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402
from shrine2.structure import structure_parameter  # noqa: E402

N_AVG, T_BIN, T_WINDOW = 8, 2, (-0.3, 0.8)  # display only


def waterfall(D, freq, time):
    """A dedispersed dynamic spectrum averaged for display, with frequency increasing upwards.

    Args:
        D: dynamic spectrum, shape (n_chan, N).
        freq, time: channel centres (MHz) and sample times (ms).
    Returns:
        W, shape (n_chan // N_AVG, n_t), and the averaged channel centres.
    """
    tw = np.flatnonzero((time >= T_WINDOW[0]) & (time <= T_WINDOW[1]))
    tw = tw[: (tw.size // T_BIN) * T_BIN]
    nf = (freq.size // N_AVG) * N_AVG
    W = D[:nf, tw].reshape(-1, N_AVG, tw.size).mean(axis=1)
    W = W.reshape(nf // N_AVG, -1, T_BIN).mean(axis=2)
    f_binned = freq[:nf].reshape(-1, N_AVG).mean(axis=1)
    if f_binned[0] > f_binned[-1]:
        W = W[::-1]
        f_binned = f_binned[::-1]
    return W, f_binned


def main():
    ex = sims.example()
    p = ex["pilot"]
    kc, d0, d, P = p["kc"], p["d0"], ex["d"], ex["P"]
    freq, time, dt_nu = ex["freq"], ex["time"], ex["dt_nu"]
    X = ex["I0"] + ex["noise"]

    # S23: the interval about the highest trial DM.
    j_max = int(np.argmax(structure_parameter(P, kc)))
    s23_lo, s23_hi, at_edge = sims.s23_interval(P, d, kc)
    assert not at_edge, "S23's interval reaches the end of the trial DMs"
    s23 = (s23_lo, d[j_max], s23_hi)

    # This work: d_fit -+ sigma_DM.
    res = dm_uncertainty(P, d, kc, dt_nu)
    ours = (res["d_fit"] - res["sig_DM"], res["d_fit"], res["d_fit"] + res["sig_DM"])

    # The delay between the band edges per unit DM, and what each offset introduces.
    band_delay = sims.K_DM * ((freq.min() / 1e3) ** -2 - (freq.max() / 1e3) ** -2)  # ms per pc cm^-3
    dt = time[1] - time[0]
    info = dict(seed=sims.EXAMPLE_SEED, kc=kc, d0=d0, s23=list(map(float, s23)), ours=list(map(float, ours)),
                sig_DM=float(res["sig_DM"]), window=[res["d_lo"], res["d_hi"]],
                sweep_ms=dict(s23=[band_delay * (s23[1] - s23[0]), band_delay * (s23[2] - s23[1])],
                              ours=band_delay * res["sig_DM"]),
                sweep_samples_ours=float(band_delay * res["sig_DM"] / dt),
                s23_halfwidth=float(0.5 * (s23[2] - s23[0])),
                ratio=float(0.5 * (s23[2] - s23[0]) / res["sig_DM"]))
    print(json.dumps(info, indent=1))

    rows = [("S23", s23), ("This work", ours)]
    images = []
    for _, dms in rows:
        row_images = []
        for dm in dms:
            row_images.append(waterfall(sims.dedisperse_spectrum(X, dm, dt_nu), freq, time))
        images.append(row_images)
    all_values = []
    for row_images in images:
        for W, _ in row_images:
            all_values.append(W.ravel())
    vmin, vmax = np.percentile(np.concatenate(all_values), [5, 99])

    # The top quarter of the band, and the time axis of the averaged samples.
    f_axis = images[0][0][1]
    top = f_axis >= f_axis.min() + 0.75 * (f_axis.max() - f_axis.min())
    f_top = f_axis[top].mean()
    tw = np.flatnonzero((time >= T_WINDOW[0]) & (time <= T_WINDOW[1]))
    tw = tw[: (tw.size // T_BIN) * T_BIN]
    t_axis = time[tw].reshape(-1, T_BIN).mean(axis=1)

    def leading_edge(W):
        """Where the 5-sample running mean of the top quarter's profile first reaches 10% of its peak,
        stepping back from the peak."""
        prof = np.convolve(W[top].sum(axis=0), np.ones(5) / 5, mode="same")
        i = int(np.argmax(prof))
        peak = prof[i]
        while i > 0 and prof[i - 1] >= 0.1 * peak:
            i -= 1
        return t_axis[i]

    def top_band_shift(d_opt, dm):
        """Change in the top quarter's dispersion delay between the optimum and trial DM dm, ms."""
        return sims.K_DM * (d_opt - dm) * ((f_top / 1e3) ** -2 - (sims.F_REF / 1e3) ** -2)

    set_pub_col(2)
    set_pub_style(use_latex=False)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.11, wspace=0.06, hspace=0.12)
    for r, (name, dms) in enumerate(rows):
        t_lead = leading_edge(images[r][1][0])
        for j in range(3):
            W, f_binned = images[r][j]
            ax = axs[r, j]
            ax.imshow(W, aspect="auto", origin="lower", cmap="plasma", vmin=vmin, vmax=vmax,
                      extent=[t_axis[0], t_axis[-1], f_binned[0], f_binned[-1]], interpolation="nearest")
            ax.tick_params(axis="both", which="both", direction="in", color="white", labelcolor="black")
            ax.axvline(t_lead + top_band_shift(dms[1], dms[j]), color="white", lw=0.7, alpha=0.8)
            ax.text(0.98, 0.97, rf"$d-d_0={dms[j] - d0:+.4f}$ pc cm$^{{-3}}$", transform=ax.transAxes,
                    ha="right", va="top", color="white", fontsize=7,
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.35, pad=2.0))
            if r == 1:
                ax.set_xlabel("Time [ms]")
            if j == 0:
                ax.set_ylabel(f"{name}\nFrequency [MHz]")
            if r == 0:
                ax.set_title(["Lower bound", "Optimum", "Upper bound"][j], fontsize=8)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_s23_range.pdf"), dpi=600, fig=fig)
    json.dump(info, open(os.path.join(HERE, "results", "fig_s23_range.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
