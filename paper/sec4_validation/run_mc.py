"""Monte Carlo validation of sigma_S and sigma_DM (Section 4), with k_c fixed.

For one burst and S/N: the pilot (k_c, d_0, the predicted sigma_DM and the trial DMs; sims.pilot),
then for each noise realisation:
    - the quoted d_fit and sigma_DM from shrine2.dm_uncertainty (Section 3.3), with sigma_S,
      Sp_max and the trial DM d_max where Sp peaks;
    - Sp at the trial DM nearest d_0 (its scatter is the Monte Carlo sigma of Sp);
    - S23's relative uncertainty of Sp at d_max (Eq. s23rel);
    - the half-width of the sigma_S crossing range (Section 3.1);
    - d_par, the vertex of the same fit to ||a|| + e_par, i.e. without the second-order term of
      Eq. spexp (the noise is known here);
    - for the first N_S23 realisations, S23's DM interval (Eq. s23int) on wider trial DMs.

Writes results/mc_<tag>_snr<snr>.npz: one value per realisation (nan where the fit failed) and
the pilot as JSON in "meta".

usage: python run_mc.py tag snr [--n_real N]
       tag is single or double; snr one of 15, 25, 40, 70, 120, 200 (the seeds are in sims.CASES)
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sims  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402
from shrine2.fit import fit_parabola  # noqa: E402

N_S23 = 200  # realisations for which S23's interval is found (it needs wider trial DMs)


def s23_grid(d0, sig_DM, d1):
    """Wider trial DMs for S23's interval: 161 trial DMs centred on d_0, spaced by the smaller of
    sigma_DM/2 and 300/80 one-sample steps.

    Args:
        d0, sig_DM: from the pilot.
        d1: one-sample DM step.
    Returns:
        (start, step, n) for sims.dedisperse_grid.
    """
    step = min(0.5 * sig_DM, 300 * d1 / 80)
    return d0 - 80 * step, step, 161


def main():
    args = get_args()
    tag, snr = args.tag, args.snr
    n_real, seed = sims.CASES[(tag, snr)]
    if args.n_real is not None:
        n_real = args.n_real

    I0, freq, time_ms = sims.load_burst(tag)
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)
    rng = np.random.default_rng(seed)
    p = sims.pilot(I0, dt_nu, sc, rng)
    kc = p["kc"]
    print(f"{tag} S/N {snr}: k_c {kc}, d0 {p['d0']:.5f}, predicted sigma_DM {p['sig_DM']:.5f}, "
          f"{p['n']} trial DMs, step {p['step']:.3g}", flush=True)

    # The trial DMs and the noiseless profiles on them; the trial DM nearest d_0.
    d = p["start"] + p["step"] * np.arange(p["n"])
    P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)
    j0 = int(np.argmin(np.abs(d - p["d0"])))

    # The noiseless structure vectors a(d_j) = D1 S s and their directions a_hat, for e_par.
    A0 = sims.structure_vectors(P0, kc)
    norm_a = np.linalg.norm(A0, axis=1)
    a_hat = A0 / norm_a[:, None]

    # Wider trial DMs for S23's interval, and the noiseless profiles on them.
    wide = s23_grid(p["d0"], p["sig_DM"], p["d1"])
    d_wide = wide[0] + wide[1] * np.arange(wide[2])
    P0_wide = sims.dedisperse_grid(I0, *wide, dt_nu)

    keys = ["d_fit", "sig_DM", "c2", "c2_err", "n_win", "d_lo", "d_hi", "sig_S", "Sp_max",
            "d_max", "Sp_d0", "s23_rel", "h_cross", "d_par", "s23_lo", "s23_hi", "s23_edge"]
    out = {}
    for k in keys:
        out[k] = np.full(n_real, np.nan)

    t0 = time.time()
    for r in range(n_real):
        noise = rng.normal(0, sc, I0.shape)
        Pn = sims.dedisperse_grid(noise, p["start"], p["step"], p["n"], dt_nu)
        P = P0 + Pn

        # Section 3.3, from the data alone.
        res = dm_uncertainty(P, d, kc, dt_nu)
        j_max = res["j_max"]
        out["sig_S"][r] = res["sig_S"]
        out["Sp_max"][r] = res["Sp_max"]
        out["d_max"][r] = res["d_max"]
        out["Sp_d0"][r] = res["Sp"][j0]
        out["s23_rel"][r] = sims.s23_relative_uncertainty(P[j_max], kc)

        # The sigma_S crossing range (Section 3.1).
        ends = sims.crossings(res["Sp"], d, j_max, res["Sp_max"] - res["sig_S"])
        if ends is not None:
            out["h_cross"][r] = 0.5 * (ends[1] - ends[0])

        if res["ok"]:
            for k in ["d_fit", "sig_DM", "n_win", "d_lo", "d_hi", "c2_err"]:
                out[k][r] = res[k]
            out["c2"][r] = res["c"][2]

            # Without the second-order term: ||a|| + e_par, fitted over the same window.
            e_par = np.sum(a_hat * sims.structure_vectors(Pn, kc), axis=1)
            win = res["win"]
            fit = fit_parabola(d[win], norm_a[win] + e_par[win])
            if fit is not None:
                out["d_par"][r] = fit[0]

        # S23's interval on the wider trial DMs.
        if r < N_S23:
            P_wide = P0_wide + sims.dedisperse_grid(noise, *wide, dt_nu)
            lo, hi, at_edge = sims.s23_interval(P_wide, d_wide, kc)
            out["s23_lo"][r], out["s23_hi"][r], out["s23_edge"][r] = lo, hi, at_edge

        if (r + 1) % 100 == 0:
            print(f"  {r + 1}/{n_real}, {time.time() - t0:.0f} s", flush=True)

    meta = dict(tag=tag, snr=snr, n_real=n_real, seed=seed, sc=sc, kc=kc, d0=p["d0"],
                sig_DM_pred=p["sig_DM"], sig_S_pred=p["sig_S"], start=p["start"], step=p["step"],
                n=p["n"], d1=p["d1"], j0=j0, wide=list(wide), runtime=time.time() - t0)
    os.makedirs(sims.VALIDATION_RESULTS, exist_ok=True)
    path = sims.results_path(f"mc_{tag}_snr{snr}.npz")
    np.savez_compressed(path, meta=json.dumps(meta), **out)
    print(f"wrote {path}", flush=True)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag", choices=["single", "double"])
    parser.add_argument("snr", type=int, choices=sims.SNRS)
    parser.add_argument("--n_real", type=int, default=None,
                        help="number of realisations (default: the paper's, in sims.CASES)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
