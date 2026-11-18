"""Appendix B check (sec:covmc): V_jl = Cov[e_par(d_j), e_par(d_l)] from Eqs. covprof and cov
(shrine2.covariance.noise_covariance) against the covariance of e_par over noise-only realisations.

The burst is the one-component FIRES burst at S/N = 40, with k_c and d_0 from the pilot of that case
in Section 4 (sims.pilot, the same seed). The trial DMs span the fit window of the noiseless burst
with the true sigma_t^2 (sims.noiseless_fit); the reference trial DM d_ref, where sigma_t^2 is set
(d_max in the method), is d_0. The coefficients w(t; d_j) are those of the noiseless burst at each
trial DM (Eq. ahat), so they are the same in every realisation.

Each realisation draws Gaussian noise in every channel of the dynamic spectrum at d_ref, with
variance s_nu sigma_t^2 at sample t, dedisperses it at every trial DM as the simulations do (each
channel shifted by (d_j - d_ref) dt_nu samples with a Fourier phase ramp, circularly, then summed
over channels) and forms e_par(d_j) = w(d_j) n(d_j). The prediction uses the same w(d_j), sigma_t^2,
s_nu and shifts, and never sees the noise realisations.

Two cases:
    white      sigma_t^2 constant, equal channels (s_nu = 1/N_nu), 41 evenly spaced trial DMs;
    selfnoise  sigma_t^2 rising by a factor of 5 at the burst's peak, following the noiseless
               frequency-summed profile at d_ref, in every channel in proportion to s_nu; s_nu drawn
               uniformly between 0.5 and 1.5 and normalised to sum to 1; 40 trial DMs drawn
               uniformly over the same range, plus d_ref.

Compares, Monte Carlo against prediction: the correlation of e_par(d_j) with e_par(d_ref); the
standard deviation of e_par(d_j), absolutely and relative to its value at d_ref (where the prediction
is exact, so the absolute comparison there tests only the sampling); and std[c1] and std[c2], for a
parabola fitted over all the trial DMs about d_ref (Eqs. lsq, varc1). Also how much the predicted
std[c1] depends on the change of std[e_par] across the trial DMs: std[c1] with the standard deviation
held at its value at d_ref (the correlations unchanged), over the full prediction.

Writes results/cov_mc_<case>.npz (d, d_ref, j_ref, V, V_mc and the numbers) and
results/cov_mc_numbers.json (both cases, after the second).

usage: python run_cov_mc.py {white,selfnoise} [--n_real 10000]
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from shrine2.covariance import noise_covariance  # noqa: E402
from shrine2.structure import coefficients  # noqa: E402

TAG, SNR = "single", 40
N_DM = 41
SELF_NOISE = 4.0  # sigma_t^2 rises by 1 + SELF_NOISE at the burst's peak (selfnoise case)
SEEDS = dict(white=401, selfnoise=402)
RESULTS = os.path.join(HERE, "results")


def setup():
    """The burst, its channel delays, the noise level and the pilot of the Section 4 case.

    Returns:
        I0, dt_nu, sig2 (the true sigma_t^2 of the profile, constant), kc, d0, and the noiseless
        fit window as (d_lo, d_hi).
    """
    I0, freq, time_ms = sims.load_burst(TAG)
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, SNR)
    _, seed = sims.CASES[(TAG, SNR)]
    p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed))
    d = p["start"] + p["step"] * np.arange(p["n"])
    fit = sims.noiseless_fit(sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu), d, p["kc"],
                             dt_nu, p["sig2"])
    win = fit["win"]
    return I0, dt_nu, p["sig2"], p["kc"], p["d0"], (d[win][0], d[win][-1])


def trial_dms(case, d0, d_lo, d_hi, rng):
    """The trial DMs and the index of d_ref = d_0 among them.

    white: N_DM evenly spaced over about [d_lo, d_hi], placed so that d_0 is one of them.
    selfnoise: N_DM - 1 drawn uniformly over [d_lo, d_hi], plus d_0.
    """
    if case == "white":
        step = (d_hi - d_lo) / (N_DM - 1)
        j0 = int(round((d0 - d_lo) / step))
        d = d0 + step * (np.arange(N_DM) - j0)
    else:
        d = np.sort(np.append(rng.uniform(d_lo, d_hi, N_DM - 1), d0))
    j_ref = int(np.argmin(np.abs(d - d0)))
    return d, j_ref


def main():
    args = get_args()
    case = args.case
    rng = np.random.default_rng(SEEDS[case])
    t0 = time.time()

    I0, dt_nu, sig2_white, kc, d0, (d_lo, d_hi) = setup()
    n_chan, N = I0.shape
    d, j_ref = trial_dms(case, d0, d_lo, d_hi, rng)
    d_ref = d[j_ref]

    # w(t; d_j) of the noiseless burst at each trial DM (Eq. ahat).
    P0 = sims.dedisperse(I0, d, dt_nu)
    W = coefficients(P0, kc)

    # sigma_t^2 at d_ref and the channels' shares s_nu.
    if case == "white":
        sig2 = sig2_white.copy()
        share = np.full(n_chan, 1.0 / n_chan)
    else:
        envelope = P0[j_ref]
        sig2 = sig2_white * (1 + SELF_NOISE * envelope / envelope.max())
        share = rng.uniform(0.5, 1.5, n_chan)
        share = share / share.sum()

    # The Monte Carlo: noise at d_ref with variance s_nu sigma_t^2 in channel nu, shifted by
    # (d_j - d_ref) dt_nu samples in each channel (Fourier phase ramp, as sims.dedisperse), summed over
    # channels, and e_par(d_j) = w(d_j) n(d_j).
    f = np.fft.rfftfreq(N)
    ramps = np.exp(2j * np.pi * f[None, None, :] * ((d - d_ref)[:, None] * dt_nu[None, :])[:, :, None])
    channel_sd = np.sqrt(share[:, None] * sig2[None, :])
    e = np.empty((args.n_real, d.size))
    for r in range(args.n_real):
        noise = rng.normal(size=(n_chan, N)) * channel_sd
        profiles = np.fft.irfft(np.einsum("jcf,cf->jf", ramps, np.fft.rfft(noise, axis=1)), N, axis=1)
        e[r] = np.sum(W * profiles, axis=1)
    V_mc = np.cov(e.T)

    # The prediction (Eqs. covprof and cov).
    V = noise_covariance(W, sig2, np.outer(d - d_ref, dt_nu), share)

    # Correlations with d_ref and their sampling errors, (1 - rho^2)/sqrt(n).
    sd, sd_mc = np.sqrt(np.diag(V)), np.sqrt(np.diag(V_mc))
    rho = V[j_ref] / (sd * sd[j_ref])
    rho_mc = V_mc[j_ref] / (sd_mc * sd_mc[j_ref])
    rho_err = np.clip(1 - rho**2, 0, None) / np.sqrt(args.n_real)
    pull_rho = np.abs(rho_mc - rho)[rho_err > 0] / rho_err[rho_err > 0]

    # How the standard deviation changes with trial DM: the Monte Carlo ratio to d_ref over the
    # predicted one. At d_ref the prediction is exact, so dividing by it removes the sampling
    # fluctuation that all the (almost fully correlated) trial DMs share; what is left has sampling
    # error sqrt((1 - rho^2)/n) for Gaussian e_par.
    rel = (sd_mc / sd_mc[j_ref]) / (sd / sd[j_ref])
    rel_err = np.sqrt(np.clip(1 - rho**2, 0, None) / args.n_real)
    pull_rel = np.abs(rel - 1)[rel_err > 0] / rel_err[rel_err > 0]

    # std[c1] and std[c2] for a parabola over all the trial DMs about d_ref (Eqs. lsq, varc1).
    A = np.vstack([np.ones(d.size), d - d_ref, (d - d_ref) ** 2]).T
    B = np.linalg.pinv(A)
    c = e @ B.T
    ratio_c1 = float(np.sqrt(B[1] @ V @ B[1]) / c[:, 1].std(ddof=1))
    # How much std[c1] depends on the change of std[e_par] with trial DM: the predicted std[c1] with
    # the standard deviation held at its d_ref value (correlations unchanged), over the full prediction.
    V_flat = V[j_ref, j_ref] * V / np.outer(sd, sd)
    c1_flat = float(np.sqrt(B[1] @ V_flat @ B[1]) / np.sqrt(B[1] @ V @ B[1]))
    ratio_c2 = float(np.sqrt(B[2] @ V @ B[2]) / c[:, 2].std(ddof=1))
    err = float(1 / np.sqrt(2 * args.n_real))

    numbers = dict(
        case=case, n_real=args.n_real, n_dm=int(d.size), kc=int(kc), d0=float(d0), d_lo=float(d[0]),
        d_hi=float(d[-1]), self_noise_peak=float(1 + SELF_NOISE) if case == "selfnoise" else 1.0,
        rho_min=float(rho.min()), max_rho_pull=float(pull_rho.max()),
        sd_ratio_min=float((sd_mc / sd).min()), sd_ratio_max=float((sd_mc / sd).max()),
        sd_ratio_ref=float(sd_mc[j_ref] / sd[j_ref]), sd_ratio_err=err,
        rel_sd_ratio_min=float(rel.min()), rel_sd_ratio_max=float(rel.max()), max_rel_sd_pull=float(pull_rel.max()),
        rel_sd_err_max=float(rel_err.max()), std_c1_flat_over_full=c1_flat,
        ratio_c1=ratio_c1, ratio_c2=ratio_c2, ratio_err=err, runtime=time.time() - t0)
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, f"cov_mc_{case}.npz"), d=d, d_ref=d_ref, j_ref=j_ref, V=V,
                        V_mc=V_mc, numbers=json.dumps(numbers))
    print(json.dumps(numbers, indent=1))

    # Both cases in one file, once both have run.
    rows = []
    for name in SEEDS:
        path = os.path.join(RESULTS, f"cov_mc_{name}.npz")
        if os.path.exists(path):
            rows.append(json.loads(str(np.load(path)["numbers"])))
    json.dump(rows, open(os.path.join(RESULTS, "cov_mc_numbers.json"), "w"), indent=1)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case", choices=list(SEEDS))
    parser.add_argument("--n_real", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
