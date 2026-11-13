"""The choice of fit window (Section 4): narrower and wider windows than the sigma_S window.

The same bursts, pilot, seeds and noise realisations as run_mc.py (the first N_REAL). For each
realisation: the quoted fit over the sigma_S window (shrine2.dm_uncertainty), giving d_fit_1 and
sigma_1; then, for each k in KS, one fit over the trial DMs with |d - d_fit_1| <= k sigma_1, with
the same sigma_t^2 (from the profile at d_max) and the same propagation (sims.fit_over).

Writes results/window_<tag>_snr<snr>.npz.

usage: python run_window.py tag snr [--n_real 300]
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

KS = [1, 1.5, 2, 2.5, 3, 4, 6]  # window half-widths, in units of the quoted sigma_DM
N_REAL = 300


def main():
    args = get_args()
    tag, snr = args.tag, args.snr
    _, seed = sims.CASES[(tag, snr)]
    n_real = args.n_real

    I0, freq, time_ms = sims.load_burst(tag)
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)
    rng = np.random.default_rng(seed)
    p = sims.pilot(I0, dt_nu, sc, rng)
    kc = p["kc"]
    d = p["start"] + p["step"] * np.arange(p["n"])
    P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)

    out = dict(d1=np.full(n_real, np.nan), s1=np.full(n_real, np.nan),
               h1=np.full(n_real, np.nan), d_fit=np.full((n_real, len(KS)), np.nan),
               sig_DM=np.full((n_real, len(KS)), np.nan))
    t0 = time.time()
    for r in range(n_real):
        noise = rng.normal(0, sc, I0.shape)
        P = P0 + sims.dedisperse_grid(noise, p["start"], p["step"], p["n"], dt_nu)

        # The quoted fit over the sigma_S window.
        res = dm_uncertainty(P, d, kc, dt_nu)
        if not res["ok"]:
            continue
        out["d1"][r] = res["d_fit"]
        out["s1"][r] = res["sig_DM"]
        out["h1"][r] = 0.5 * (res["d_hi"] - res["d_lo"])

        # Refits over |d - d_fit_1| <= k sigma_1, skipped if the window reaches the trial DMs' ends.
        for i, k in enumerate(KS):
            win = np.abs(d - res["d_fit"]) <= k * res["sig_DM"] + 1e-12
            if win[0] or win[-1] or win.sum() < 3:
                continue
            fit = sims.fit_over(P, d, win, res["j_max"], kc, dt_nu, res["sig2"])
            if fit is not None:
                out["d_fit"][r, i] = fit["d_fit"]
                out["sig_DM"][r, i] = fit["sig_DM"]

        if (r + 1) % 100 == 0:
            print(f"  {r + 1}/{n_real}, {time.time() - t0:.0f} s", flush=True)

    meta = dict(tag=tag, snr=snr, n_real=n_real, seed=seed, kc=kc, d0=p["d0"], ks=KS,
                runtime=time.time() - t0)
    os.makedirs(sims.VALIDATION_RESULTS, exist_ok=True)
    path = sims.results_path(f"window_{tag}_snr{snr}.npz")
    np.savez_compressed(path, meta=json.dumps(meta), **out)
    print(f"wrote {path}", flush=True)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag", choices=["single", "double"])
    parser.add_argument("snr", type=int, choices=[15, 40, 200])
    parser.add_argument("--n_real", type=int, default=N_REAL)
    return parser.parse_args()


if __name__ == "__main__":
    main()
