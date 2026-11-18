"""Monte Carlo tests (Sections 4.3 and 5.3).

Two tests, each on the noise realisations of its own seed (the self-noise-like test that was here is
now run_mc_selfnoise.py, Section 4.2):

    offcov SPECTRUM SNR The coloured noise of run_mc_coloured.py (ar1 or red), with Cov(n)
                        estimated from off-pulse data, as an observer would: for each
                        realisation, M_OFF independent off-pulse profiles of N samples with the
                        same noise, whose mean periodogram gives a circulant Cov(n) (the noise
                        of the simulations is stationary on the circle). Evaluation: est_cov.

    morph TAG SNR       The paper's procedure on a burst of different morphology: drift020, three
                        sub-bursts in separate subbands, separated by 0.2 ms (Section 5.1). White
                        noise, k_c fixed, as run_mc.py.

d_0 is d_fit of the noiseless burst with the true noise (for offcov, as in run_mc_coloured.py), and
k_c that of the white-noise pilot. The trial DMs are spaced by sigma_DM/6 of the noiseless burst
with the true noise.

Writes results/review_<test>_<tag>_snr<snr>.npz.

usage: python run_mc_review.py test tag snr [--n_real N]
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
from scipy.linalg import circulant

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import sims  # noqa: E402
from run_mc_coloured import colour, noiseless_sigma, power_spectrum  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402

M_OFF = 4  # offcov: off-pulse profiles (of N samples each) from which Cov(n) is estimated
SEEDS = {"offcov": 600, "morph": 700}  # added to the S/N for each test's seed


def run_offcov(spectrum, snr, n_real, rng):
    I0, freq, time_ms = sims.load_burst("single")
    n_chan, N = I0.shape
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)
    _, seed_w = sims.CASES[("single", snr)]
    p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed_w))
    kc = p["kc"]
    H = power_spectrum(spectrum, N)
    cov_full = n_chan * sc**2 * circulant(np.fft.irfft(H, N))

    # Trial DMs as in run_mc_coloured.py.
    grid = (p["d0"] - 120 * p["d1"], p["d1"] / 2, 481)
    for _ in range(3):
        d = grid[0] + grid[1] * np.arange(grid[2])
        sig_pred = noiseless_sigma(sims.dedisperse_grid(I0, *grid, dt_nu), d, kc, dt_nu, cov_full)
        grid = sims.validation_grid(p["d0"], sig_pred, p["d1"])
    d = grid[0] + grid[1] * np.arange(grid[2])
    P0 = sims.dedisperse_grid(I0, *grid, dt_nu)
    j0 = int(np.argmin(np.abs(d - p["d0"])))
    print(f"offcov {spectrum} S/N {snr}: k_c {kc}, d0 {p['d0']:.5f}, predicted sigma_DM "
          f"{sig_pred:.5f}, {grid[2]} trial DMs", flush=True)

    out = {"Sp_d0": np.full(n_real, np.nan), "var_ratio": np.full(n_real, np.nan)}
    for k in ["d_fit", "sig_DM", "sig_S", "c2_err"]:
        out[f"{k}_est_cov"] = np.full(n_real, np.nan)
    t0 = time.time()
    for r in range(n_real):
        noise = sc * colour(rng.normal(0, 1, I0.shape), H)
        P = P0 + sims.dedisperse_grid(noise, *grid, dt_nu)

        # Off-pulse profiles: M_OFF independent stretches of N samples, summed over channels.
        off = sc * colour(rng.normal(0, 1, (M_OFF * n_chan, N)), H)
        off = off.reshape(M_OFF, n_chan, N).sum(axis=1)
        off = off - off.mean(axis=1, keepdims=True)
        # Circulant Cov(n) from the mean periodogram (zero-mean profiles, so the k=0 term is
        # empty; it does not affect D1 S n).
        pgram = np.mean(np.abs(np.fft.fft(off, axis=1)) ** 2, axis=0) / N
        cov_est = circulant(np.real(np.fft.ifft(pgram)))
        out["var_ratio"][r] = cov_est[0, 0] / cov_full[0, 0]

        res = dm_uncertainty(P, d, kc, dt_nu, cov_n=cov_est)
        out["Sp_d0"][r] = res["Sp"][j0]
        out["sig_S_est_cov"][r] = res["sig_S"]
        if res["ok"]:
            out["d_fit_est_cov"][r] = res["d_fit"]
            out["sig_DM_est_cov"][r] = res["sig_DM"]
            out["c2_err_est_cov"][r] = res["c2_err"]
        if (r + 1) % 50 == 0:
            print(f"  {r + 1}/{n_real}, {time.time() - t0:.0f} s", flush=True)
    meta = dict(kc=kc, d0=p["d0"], sig_DM_pred=sig_pred, m_off=M_OFF, j0=j0, grid=list(grid),
                sc=sc)
    return out, meta


def run_morph(tag, snr, n_real, rng):
    I0, freq, time_ms = sims.load_burst(tag)
    n_chan, N = I0.shape
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)
    p = sims.pilot(I0, dt_nu, sc, rng)
    kc = p["kc"]
    d = p["start"] + p["step"] * np.arange(p["n"])
    P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)
    j0 = int(np.argmin(np.abs(d - p["d0"])))
    print(f"morph {tag} S/N {snr}: k_c {kc}, d0 {p['d0']:.5f}, predicted sigma_DM "
          f"{p['sig_DM']:.5f}, {p['n']} trial DMs", flush=True)
    out = {"Sp_d0": np.full(n_real, np.nan)}
    for k in ["d_fit", "sig_DM", "sig_S", "c2_err"]:
        out[f"{k}_est"] = np.full(n_real, np.nan)
    t0 = time.time()
    for r in range(n_real):
        noise = rng.normal(0, sc, I0.shape)
        P = P0 + sims.dedisperse_grid(noise, p["start"], p["step"], p["n"], dt_nu)
        res = dm_uncertainty(P, d, kc, dt_nu)
        out["Sp_d0"][r] = res["Sp"][j0]
        out["sig_S_est"][r] = res["sig_S"]
        if res["ok"]:
            out["d_fit_est"][r] = res["d_fit"]
            out["sig_DM_est"][r] = res["sig_DM"]
            out["c2_err_est"][r] = res["c2_err"]
        if (r + 1) % 100 == 0:
            print(f"  {r + 1}/{n_real}, {time.time() - t0:.0f} s", flush=True)
    meta = dict(kc=kc, d0=p["d0"], sig_DM_pred=p["sig_DM"], sig_S_pred=p["sig_S"], j0=j0,
                grid=[p["start"], p["step"], p["n"]], sc=sc)
    return out, meta


def main():
    args = get_args()
    seed = SEEDS[args.test] + args.snr
    rng = np.random.default_rng(seed)
    run = dict(offcov=run_offcov, morph=run_morph)[args.test]
    t0 = time.time()
    out, meta = run(args.tag, args.snr, args.n_real, rng)
    meta.update(test=args.test, tag=args.tag, snr=args.snr, n_real=args.n_real, seed=seed,
                runtime=time.time() - t0)
    os.makedirs(sims.VALIDATION_RESULTS, exist_ok=True)
    path = sims.results_path(f"review_{args.test}_{args.tag}_snr{args.snr}.npz")
    np.savez_compressed(path, meta=json.dumps(meta), **out)
    print(f"wrote {path}", flush=True)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("test", choices=["offcov", "morph"])
    parser.add_argument("tag", help="drift020 (morph); ar1/red (offcov)")
    parser.add_argument("snr", type=int)
    parser.add_argument("--n_real", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    main()
