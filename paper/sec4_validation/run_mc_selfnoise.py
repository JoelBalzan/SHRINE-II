"""Monte Carlo with noise whose variance follows the burst (Section 4.2).

The other Section 4 runs use noise of constant variance, so they do not test the estimate of
sigma_t^2 from the profile (Section 3.3.2) where it changes with t. Here the noise in channel nu has
variance

    sc^2 [1 + g I_nu(t)^2],

with I_nu(t) the noiseless dynamic spectrum at the true DM and sc the channel noise of run_mc.py at
the given S/N. g is one number per burst, the same for every channel and sample:

    g = 4 N_nu / max_t sum_nu I_nu(t)^2,

so that the variance of the frequency-summed profile at the true DM, N_nu sc^2 [1 + (g / N_nu)
sum_nu I_nu(t)^2], peaks at five times its off-pulse value, N_nu sc^2. g does not depend on sc, so
the ratio is five at every S/N. The noise is added to the dynamic spectrum at the true DM, i.e. it
is attached to the emission and dedispersed with it. Each channel's share of the profile variance,
[1 + g I_nu(t)^2] / sum_nu' [1 + g I_nu'(t)^2], changes with t because scattering changes the burst
across the band, which the noise model of Appendix B (fixed shares s_nu) does not describe.

This is a stress test of a time-dependent variance and of changing channel shares, not a model of
real self-noise: that scales as (1 + I_nu / I_sys)^2, with I_sys changing across the band, and its
peak ratio grows with the burst's brightness.

For each realisation:
    est   shrine2.dm_uncertainty as for real data: sigma_t^2 estimated from the profile at d_max,
          equal channel shares (the off-pulse shares);
    true  the same with the true sigma_t^2 of the profile at d_max given as cov_n.
The true sigma_t^2 at trial DM d is the exact variance of the dedispersed profile: dedispersion
applies to each channel a circular filter h_nu (the Fourier phase ramp of sims.dedisperse), so the
channel's variance becomes the circular convolution of its variance with h_nu^2.

k_c is that of the white-noise pilot (sims.pilot, the seed of run_mc.py). d_0, the predicted
sigma_DM and the trial DMs come from the noiseless burst with the true sigma_t^2 (sims.noiseless_fit),
as in the pilot: the trial DMs are spaced by sigma_DM/6 about d_0 and refined until sigma_DM changes
by less than 1%. Also records the estimate of sigma_t^2 against the true one at d_max: at the true
variance's peak, and over the samples where the true variance exceeds 1.5 times its off-pulse value.

Writes results/selfnoise_<tag>_snr<snr>.npz.

usage: python run_mc_selfnoise.py tag snr [--n_real N]
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
from shrine2.structure import structure_parameter  # noqa: E402

PEAK_RATIO = 5.0  # profile variance at its peak / off-pulse, at the true DM
HIGH = 1.5  # samples where the true variance exceeds HIGH times off-pulse, for the sigma_t^2 check
SEED_OFFSET = 2000  # the noise seeds are those of run_mc.py plus this, so the noise is new
CASES = {("single", 15): 400, ("single", 40): 400, ("single", 200): 300, ("double", 40): 300}


def gain(I0):
    """g such that the frequency-summed profile variance at the true DM peaks at PEAK_RATIO times
    its off-pulse value.

    Args:
        I0: noiseless dynamic spectrum at the true DM, shape (n_chan, N).
    Returns:
        g, in units of 1 / I0^2.
    """
    return (PEAK_RATIO - 1) * I0.shape[0] / np.max(np.sum(I0**2, axis=0))


def profile_variance(var_dyn, d, dt_nu):
    """The exact variance of the frequency-summed profile at trial DM d of independent noise with
    variance var_dyn (channel, sample) at the true DM, dedispersed as sims.dedisperse does.

    Each channel is filtered by h_nu = irfft(ramp_nu), circularly, so its variance becomes
    var_nu circularly convolved with h_nu^2; the channels are independent, so their variances add.

    Args:
        var_dyn: noise variance at the true DM, shape (n_chan, N).
        d: trial DM (offset from the true DM).
        dt_nu: channel delays per unit DM, samples, shape (n_chan,).
    Returns:
        sigma_t^2, shape (N,).
    """
    N = var_dyn.shape[1]
    f = np.fft.rfftfreq(N)
    h = np.fft.irfft(np.exp(2j * np.pi * f[None, :] * (d * dt_nu)[:, None]), N, axis=1)
    return np.fft.irfft(np.fft.rfft(var_dyn, axis=1) * np.fft.rfft(h**2, axis=1), N, axis=1).sum(axis=0)


def noiseless(I0, grid, kc, dt_nu, var_dyn):
    """d_0 and the predicted sigma_DM of the noiseless burst with the true sigma_t^2 (at the
    noiseless peak, where noiseless_fit sets it)."""
    d = grid[0] + grid[1] * np.arange(grid[2])
    P0 = sims.dedisperse_grid(I0, *grid, dt_nu)
    j_max = int(np.argmax(structure_parameter(P0, kc)))
    return sims.noiseless_fit(P0, d, kc, dt_nu, profile_variance(var_dyn, d[j_max], dt_nu))


def main():
    args = get_args()
    tag, snr = args.tag, args.snr
    n_real = args.n_real if args.n_real is not None else CASES[(tag, snr)]
    _, seed = sims.CASES[(tag, snr)]

    I0, freq, time_ms = sims.load_burst(tag)
    n_chan, N = I0.shape
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)

    # k_c from the white-noise pilot of run_mc.py, so that only the noise changes.
    p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed))
    kc, d1 = p["kc"], p["d1"]

    # The noise variance in every channel and sample, at the true DM.
    g = gain(I0)
    var_dyn = sc**2 * (1 + g * I0**2)
    off = n_chan * sc**2  # off-pulse profile variance

    # d_0, sigma_DM and the trial DMs from the noiseless burst with the true sigma_t^2, as in the
    # pilot: start half a one-sample step apart over +-200 such steps about the white-noise d_0,
    # then space by sigma_DM / 6 until sigma_DM changes by less than 1%.
    grid = (p["d0"] - 200 * d1, d1 / 2, 801)
    sig_old = np.inf
    for _ in range(10):
        pred = noiseless(I0, grid, kc, dt_nu, var_dyn)
        grid = sims.validation_grid(pred["d0"], pred["sig_DM"], d1)
        if abs(pred["sig_DM"] / sig_old - 1) < 0.01:
            break
        sig_old = pred["sig_DM"]
    d = grid[0] + grid[1] * np.arange(grid[2])
    P0 = sims.dedisperse_grid(I0, *grid, dt_nu)
    j0 = int(np.argmin(np.abs(d - pred["d0"])))
    print(f"{tag} S/N {snr}: k_c {kc}, g {g:.4g}, d0 {pred['d0']:.5f} (white {p['d0']:.5f}), predicted "
          f"sigma_DM {pred['sig_DM']:.5f} (white {p['sig_DM']:.5f}), {grid[2]} trial DMs", flush=True)

    out = dict(Sp_d0=np.full(n_real, np.nan), Sp_max=np.full(n_real, np.nan),
               d_max=np.full(n_real, np.nan), sig2_peak_ratio=np.full(n_real, np.nan),
               sig2_high_ratio=np.full(n_real, np.nan))
    for model in ["est", "true"]:
        for k in ["d_fit", "sig_DM", "sig_S"]:
            out[f"{k}_{model}"] = np.full(n_real, np.nan)

    sig2_true_at = {}  # the true sigma_t^2 at each trial DM where Sp has peaked so far
    rng = np.random.default_rng(seed + SEED_OFFSET)
    t0 = time.time()
    for r in range(n_real):
        noise = rng.normal(0, 1, I0.shape) * np.sqrt(var_dyn)
        P = P0 + sims.dedisperse_grid(noise, *grid, dt_nu)

        res = dm_uncertainty(P, d, kc, dt_nu)
        j_max = res["j_max"]
        out["Sp_d0"][r] = res["Sp"][j0]
        out["Sp_max"][r] = res["Sp_max"]
        out["d_max"][r] = res["d_max"]
        out["sig_S_est"][r] = res["sig_S"]
        if res["ok"]:
            out["d_fit_est"][r], out["sig_DM_est"][r] = res["d_fit"], res["sig_DM"]

        # The estimate of sigma_t^2 against the true one, at d_max.
        if j_max not in sig2_true_at:
            sig2_true_at[j_max] = profile_variance(var_dyn, d[j_max], dt_nu)
        sig2_true = sig2_true_at[j_max]
        t_peak = int(np.argmax(sig2_true))
        high = sig2_true > HIGH * off
        out["sig2_peak_ratio"][r] = res["sig2"][t_peak] / sig2_true[t_peak]
        out["sig2_high_ratio"][r] = np.mean(res["sig2"][high]) / np.mean(sig2_true[high])

        res = dm_uncertainty(P, d, kc, dt_nu, cov_n=sig2_true)
        out["sig_S_true"][r] = res["sig_S"]
        if res["ok"]:
            out["d_fit_true"][r], out["sig_DM_true"][r] = res["d_fit"], res["sig_DM"]

        if (r + 1) % 50 == 0:
            print(f"  {r + 1}/{n_real}, {time.time() - t0:.0f} s", flush=True)

    meta = dict(tag=tag, snr=snr, n_real=n_real, seed=seed + SEED_OFFSET, sc=sc, kc=kc, g=g,
                peak_ratio=PEAK_RATIO, d0=pred["d0"], sig_DM_pred=pred["sig_DM"],
                sig_DM_pred_white=p["sig_DM"], d0_white=p["d0"], start=grid[0], step=grid[1],
                n=grid[2], j0=j0, runtime=time.time() - t0)
    os.makedirs(sims.VALIDATION_RESULTS, exist_ok=True)
    path = sims.results_path(f"selfnoise_{tag}_snr{snr}.npz")
    np.savez_compressed(path, meta=json.dumps(meta), **out)
    print(f"wrote {path}", flush=True)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag", choices=["single", "double"])
    parser.add_argument("snr", type=int, choices=[15, 40, 200])
    parser.add_argument("--n_real", type=int, default=None,
                        help="number of realisations (default: the paper's, in CASES)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
