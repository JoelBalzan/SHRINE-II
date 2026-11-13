"""Figure 4 (fig:s23_demo, Section 2.4): the noise components of Sp(d) for the example burst.

The noise is known here, so at every trial DM we have a = D1 S s, e = D1 S n, e_par = a_hat . e
and e_perp = e - e_par a_hat. Near the peak, the figure shows ||a||, ||a + e_par a_hat||
(= ||a|| + e_par), ||a|| + c1 (d - d_0) with c1 the slope of e_par fitted over the fit window,
||a + e_perp|| and the measured Sp = ||a + e||, with the vertex of the same parabola fit (same
window) to each, and the fit window of Section 3.3 (from shrine2.dm_uncertainty).

The number quoted in Section 3.2: the linear term alone moves the peak (from that of ||a||) by
peak_shift_ratio of the shift from all of e_par.

Writes figures/fig_s23_demo.pdf and results/fig_s23_demo.json.

usage: python make_demo_fig.py [--search]
       --search repeats the choice of the example realisation: of the first 400 seeds, the one with
       e_par(d_0) closest to +sigma_S, d_fit - d_0 closest to +sigma_DM and a typical quoted
       sigma_DM (the paper uses seed sims.EXAMPLE_SEED)
"""
import json
import os
import sys
from argparse import ArgumentParser

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from plotting import IBM_PALETTE, savefig, set_pub_col, set_pub_style  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402
from shrine2.fit import fit_parabola  # noqa: E402

N_SEARCH = 400


def components(noise, ex, a_hat):
    """The noise profiles at the trial DMs, e = D1 S n, and its parts along and across a_hat.

    Args:
        noise: noise dynamic spectrum, shape (n_chan, N).
        ex: sims.example().
        a_hat: unit structure vectors of the noiseless burst, shape (n_d, N - 1).
    Returns:
        Pn: the noise profiles, shape (n_d, N); e: D1 S n, shape (n_d, N - 1);
        e_par: shape (n_d,); norm_e_perp: ||e_perp||, shape (n_d,).
    """
    d = ex["d"]
    kc = ex["pilot"]["kc"]
    Pn = sims.dedisperse_grid(noise, d[0], d[1] - d[0], d.size, ex["dt_nu"])
    e = sims.structure_vectors(Pn, kc)
    e_par = np.sum(a_hat * e, axis=1)
    e_perp = e - e_par[:, None] * a_hat
    return Pn, e, e_par, np.linalg.norm(e_perp, axis=1)


def search_seed(ex, a_hat, j0):
    """Repeat the choice of the example realisation (see the module docstring)."""
    p = ex["pilot"]
    best_score, best_seed = np.inf, None
    for seed in range(N_SEARCH):
        noise = np.random.default_rng(seed).normal(0, ex["sc"], ex["I0"].shape)
        Pn, _, e_par, _ = components(noise, ex, a_hat)
        res = dm_uncertainty(ex["P0"] + Pn, ex["d"], p["kc"], ex["dt_nu"])
        if not res["ok"]:
            continue
        score = ((e_par[j0] / p["sig_S"] - 1) ** 2 + ((res["d_fit"] - p["d0"]) / p["sig_DM"] - 1) ** 2
                 + (res["sig_DM"] / p["sig_DM"] - 1) ** 2)
        if score < best_score:
            best_score, best_seed = score, seed
    return best_seed


def main():
    args = get_args()
    BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE
    ex = sims.example()
    p = ex["pilot"]
    kc, d0, d = p["kc"], p["d0"], ex["d"]

    # The noiseless structure vectors and their directions.
    A = sims.structure_vectors(ex["P0"], kc)
    norm_a = np.linalg.norm(A, axis=1)
    a_hat = A / norm_a[:, None]
    j0 = int(np.argmin(np.abs(d - d0)))

    seed = sims.EXAMPLE_SEED
    if args.search:
        seed = search_seed(ex, a_hat, j0)
        print(f"best seed of the first {N_SEARCH}: {seed}")
    noise = np.random.default_rng(seed).normal(0, ex["sc"], ex["I0"].shape)
    Pn, e, e_par, norm_e_perp = components(noise, ex, a_hat)

    # The curves: measured, with e_par only, with e_perp only.
    Sp = np.linalg.norm(A + e, axis=1)
    S_par = norm_a + e_par
    S_perp = np.sqrt(norm_a**2 + norm_e_perp**2)

    # Section 3.3 on the data: the fit window, d_fit and sigma_DM.
    res = dm_uncertainty(ex["P0"] + Pn, d, kc, ex["dt_nu"])
    win = res["win"]

    # The slope of e_par over the window, c1 = b_1 e_par (Eq. lsq), and ||a|| tilted by it.
    slope = float(res["B"][1] @ e_par[win])
    S_tilt = norm_a + slope * (d - d0)

    # The vertex of the same parabola fit (same window) applied to each curve.
    curves = dict(a=norm_a, par=S_par, tilt=S_tilt, perp=S_perp, meas=Sp)
    peaks = {}
    peaks_json = {}
    for name in curves:
        d_v, c, _ = fit_parabola(d[win], curves[name][win])
        peaks[name] = (d_v, c[0])
        peaks_json[name] = [float(d_v), float(c[0])]
    shift_ratio = (peaks["tilt"][0] - peaks["a"][0]) / (peaks["par"][0] - peaks["a"][0])

    # S23 at the highest trial DM, for the record.
    s23_lo, s23_hi, at_edge = sims.s23_interval(ex["P0"] + Pn, d, kc)

    info = dict(seed=seed, kc=kc, d0=d0, d_fit=res["d_fit"], sig_DM=res["sig_DM"],
                sig_DM_pred=p["sig_DM"], sig_S=res["sig_S"], sig_S_pred=p["sig_S"],
                Sp_max=res["Sp_max"], e_par_d0=float(e_par[j0]),
                e_perp_bias=float(S_perp[j0] - norm_a[j0]), window=[res["d_lo"], res["d_hi"]],
                n_win=res["n_win"], slope_fit=slope, peak_shift_ratio=float(shift_ratio),
                peaks=peaks_json,
                s23_interval=[float(s23_lo), float(s23_hi)], s23_at_edge=bool(at_edge))
    print(json.dumps(info, indent=1))

    # The figure: Sp(d) near the peak.
    set_pub_col(1)
    set_pub_style(use_latex=False)
    plt.rcParams["figure.constrained_layout.use"] = False
    X = d - d0
    x_fit = res["d_fit"] - d0
    centre = 0.5 * (res["d_lo"] + res["d_hi"]) - d0
    half = 0.5 * (res["d_hi"] - res["d_lo"])
    sig_S = res["sig_S"]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15)
    ax.axvspan(centre - half, centre + half, color=PURP, alpha=0.10, lw=0, zorder=0)
    ax.plot(X, norm_a, color="k", lw=1.3, label=r"$\|\mathbf{a}\|$", zorder=3)
    ax.plot(X, S_par, color=BLUE, lw=1.1, label=r"$\|\mathbf{a}+e_\parallel\hat{\mathbf{a}}\|$",
            zorder=3)
    ax.plot(X, S_tilt, color=BLUE, lw=1.1, ls=(0, (4, 2)),
            label=r"$\|\mathbf{a}\|+e_\parallel'\,(d-d_0)$", zorder=3)
    ax.plot(X, S_perp, color=YELL, lw=1.4, label=r"$\|\mathbf{a}+\mathbf{e}_\perp\|$", zorder=3)
    ax.plot(X, Sp, color=ORNG, lw=1.1, ls="--",
            label=r"Measured, $\mathcal{S}=\|\mathbf{a}+\mathbf{e}\|$", zorder=4)
    for name, colour in [("a", "k"), ("par", BLUE), ("perp", YELL), ("meas", ORNG)]:
        ax.plot(peaks[name][0] - d0, peaks[name][1], "o", ms=3.5, color=colour, mec="white",
                mew=0.7, zorder=6)
    ax.plot(peaks["tilt"][0] - d0, peaks["tilt"][1], "o", ms=3.5, color="white", mec=BLUE,
            mew=1.0, zorder=6)
    ax.axvline(0, color="k", lw=0.6, ls=":", zorder=2)
    ax.axvline(x_fit, color=PURP, lw=0.6, ls=":", zorder=2)
    y_top = norm_a.max() + 3.5 * sig_S
    ax.text(0, y_top - 0.35 * sig_S, r"$d_0$", ha="right", va="top", fontsize=7)
    ax.text(x_fit + 0.06 * half, y_top - 0.35 * sig_S, r"$d_{\rm fit}$", ha="left", va="top",
            fontsize=7, color=PURP)
    ax.set_xlim(centre - 1.6 * half, centre + 1.6 * half)
    ax.set_ylim(norm_a.max() - 8.5 * sig_S, norm_a.max() + 3.5 * sig_S)
    ax.set_xlabel(r"$d-d_0$ [pc cm$^{-3}$]")
    ax.set_ylabel(r"$\mathcal{S}(d)$")
    ax.legend(loc="lower center", fontsize=6, ncol=2, columnspacing=0.8, handlelength=1.8)
    ax.grid(True, lw=0.5, alpha=0.5)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    savefig(os.path.join(HERE, "figures", "fig_s23_demo.pdf"), dpi=600, fig=fig)
    json.dump(info, open(os.path.join(HERE, "results", "fig_s23_demo.json"), "w"), indent=1)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--search", action="store_true",
                        help=f"choose the realisation again from the first {N_SEARCH} seeds")
    return parser.parse_args()


if __name__ == "__main__":
    main()
