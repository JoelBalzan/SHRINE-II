"""Monte Carlo with coloured noise: does sigma_DM hold when the noise is correlated between samples?

The derivation does not need white noise: sigma_S^2 = w Cov(n) w^T (Eq. sigSgen) and V (Eq. cov)
hold for any Cov(n) shared by the channels in the proportions s_nu. What assumes white noise is the
estimate of the noise from the profile (Section 3.3.2). This script repeats the Monte Carlo of
run_mc.py for the one-component burst with stationary coloured noise of known spectrum in every
channel, and compares three ways of evaluating sigma_S and sigma_DM on the same realisations:

    white   the paper's procedure as it stands: sigma_t^2 from the profile's high DCT modes
            (shrine2.dm_uncertainty);
    diag    the true noise variance per sample, but ignoring the correlation between samples
            (Cov(n) taken as diagonal);
    full    the true Cov(n), a full N x N matrix: sigma_S^2 = w Cov(n) w^T and V from
            shrine2.covariance.noise_covariance with cov_n = Cov(n).
diag and full use shrine2.dm_uncertainty with cov_n given (Sec. 5.3 of the paper).

Each sets its own fit window (from its own sigma_S), so each has its own d_fit and its own Monte
Carlo scatter, against which its median sigma_DM is compared; summarise_coloured.py also compares
each realisation's sigma_DM with its own d_fit - d_0. For the full model, the script also
records d_par, the vertex of the same fit to ||a|| + e_par (the noise is known), i.e. without the
second-order term of Eq. spexp, which grows with sqrt(<||e||^2>)/||a||.

The coloured noise is generated in each channel by filtering unit white noise with the square root
of a power spectrum H(f) (f in cycles per sample), circularly, so that it is stationary on the
circle and the circular dedispersion of the simulations leaves its covariance unchanged:
    ar1   H(f) = (1 - rho^2) / (1 + rho^2 - 2 rho cos 2 pi f), rho = 0.5: correlation 0.5 between
          adjacent samples, falling by half every sample; the variance per sample is that of the
          white noise of run_mc.py;
    red   H(f) = 1 + (f_knee / f)^2 (and 1 at f = 0): the white noise of run_mc.py plus a red
          component that equals it at f_knee = 0.0113 cycles per sample (DCT mode k ~ 14, half the
          k_c of S/N = 40) and falls as f^-2 above it.
S/N is the nominal one, of the white noise in run_mc.py (FIRES's level), for both spectra. k_c and
d_0 are those of the white-noise pilot (sims.pilot), so only the noise changes; the trial DMs are
spaced by sigma_DM/6 of the noiseless burst with the true Cov(n). Also records
sqrt(<||e||^2>)/||a|| at d_0 with the true Cov(n), the ratio that the first-order expansion needs to
be small (Section 3.4): coloured noise of the same variance raises it.

Writes results/coloured_<spectrum>_snr<snr>.npz.

usage: python run_mc_coloured.py spectrum snr [--n_real 500]
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
from scipy.linalg import circulant

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sims  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402
from shrine2.fit import fit_parabola, fit_window  # noqa: E402
from shrine2.structure import coefficients, forward_difference_matrix, smooth, structure_parameter  # noqa: E402

RHO = 0.5  # ar1: correlation between adjacent samples
F_KNEE = 0.0113  # red: frequency (cycles per sample) at which the red component equals the white
SEED_OFFSET = 1000  # the noise seeds are those of run_mc.py plus this, so the noise is new
MODELS = ["white", "diag", "full"]


def power_spectrum(spectrum, N):
    """H(f) at the rfft frequencies f = k/N, k = 0 ... N//2, for one channel's noise.

    Args:
        spectrum: "ar1" or "red".
        N: samples per profile.
    Returns:
        H, shape (N//2 + 1,), scaled so that the white part (red) or the whole spectrum (ar1) has
        unit variance per sample.
    """
    f = np.fft.rfftfreq(N)
    if spectrum == "ar1":
        H = (1 - RHO**2) / (1 + RHO**2 - 2 * RHO * np.cos(2 * np.pi * f))
        # Unit variance per sample: c(0), the mean of H over the full circle, is 1.
        H = H / np.fft.irfft(H, N)[0]
    else:
        H = np.ones_like(f)
        H[1:] += (F_KNEE / f[1:]) ** 2
    return H


def colour(white, H):
    """Filter white noise in each channel by sqrt(H), circularly.

    Args:
        white: unit white noise, shape (n_chan, N).
        H: power spectrum at the rfft frequencies.
    Returns:
        coloured noise, shape (n_chan, N), with circulant covariance whose first row is irfft(H).
    """
    N = white.shape[1]
    return np.fft.irfft(np.fft.rfft(white, axis=1) * np.sqrt(H), N, axis=1)


def noiseless_sigma(P0, d, kc, dt_nu, cov_n):
    """d_0's window and predicted sigma_DM for the noiseless burst with the true Cov(n) (as
    sims.noiseless_fit, with the full covariance in sigma_S and V).

    Returns:
        the predicted sigma_DM.
    """
    norm_a = structure_parameter(P0, kc)
    j_max = int(np.argmax(norm_a))
    w = coefficients(P0[j_max], kc)
    win = fit_window(norm_a, np.sqrt(w @ cov_n @ w), j_max)
    return sims.fit_over(P0, d, win, j_max, kc, dt_nu, cov_n)["sig_DM"]


def main():
    args = get_args()
    spectrum, snr = args.spectrum, args.snr
    _, seed = sims.CASES[("single", snr)]

    I0, freq, time_ms = sims.load_burst("single")
    n_chan, N = I0.shape
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)

    # k_c and d_0 from the white-noise pilot of run_mc.py, so that only the noise changes.
    p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed))
    kc = p["kc"]

    # The true Cov(n) of the frequency-summed profile: n_chan channels, each with circulant
    # covariance sc^2 circulant(irfft(H)) (the same at every trial DM, since the noise is
    # stationary on the circle and dedispersion shifts it circularly).
    H = power_spectrum(spectrum, N)
    autocov = np.fft.irfft(H, N)
    cov_full = n_chan * sc**2 * circulant(autocov)
    cov_diag = np.diag(cov_full).copy()

    # Trial DMs: spaced by sigma_DM/6 of the noiseless burst with the true Cov(n). Start from
    # trial DMs half a one-sample step apart over +-120 such steps about d_0 (wide enough for the
    # window of the noiseless burst), then refine twice.
    grid = (p["d0"] - 120 * p["d1"], p["d1"] / 2, 481)
    for _ in range(3):
        d = grid[0] + grid[1] * np.arange(grid[2])
        sig_pred = noiseless_sigma(sims.dedisperse_grid(I0, *grid, dt_nu), d, kc, dt_nu, cov_full)
        grid = sims.validation_grid(p["d0"], sig_pred, p["d1"])
    d = grid[0] + grid[1] * np.arange(grid[2])
    P0 = sims.dedisperse_grid(I0, *grid, dt_nu)
    j0 = int(np.argmin(np.abs(d - p["d0"])))

    # sqrt(<||e||^2>)/||a|| at d_0 with the true Cov(n): <||e||^2> = tr(D1 S Cov(n) S D1^T), with
    # S as a matrix (smooth() applied to the rows of the identity; S is symmetric).
    D1S = forward_difference_matrix(N) @ smooth(np.eye(N), kc)
    s0 = sims.dedisperse(I0, [p["d0"]], dt_nu)[0]
    e_over_a = np.sqrt(np.trace(D1S @ cov_full @ D1S.T)) / np.linalg.norm(D1S @ s0)
    e_over_a_white = np.sqrt(n_chan * sc**2 * np.sum(D1S**2)) / np.linalg.norm(D1S @ s0)
    print(f"{spectrum} S/N {snr}: k_c {kc}, d0 {p['d0']:.5f}, predicted sigma_DM {sig_pred:.5f} "
          f"(white-noise pilot {p['sig_DM']:.5f}), {grid[2]} trial DMs; sqrt(<||e||^2>)/||a|| "
          f"{e_over_a:.2f} (white noise {e_over_a_white:.2f})", flush=True)

    # The noiseless structure vectors and their directions, for e_par.
    A0 = sims.structure_vectors(P0, kc)
    norm_a = np.linalg.norm(A0, axis=1)
    a_hat = A0 / norm_a[:, None]

    out = dict(Sp_d0=np.full(args.n_real, np.nan), Sp_max=np.full(args.n_real, np.nan),
               d_par_full=np.full(args.n_real, np.nan))
    for model in MODELS:
        for k in ["d_fit", "sig_DM", "sig_S", "n_win"]:
            out[f"{k}_{model}"] = np.full(args.n_real, np.nan)

    rng = np.random.default_rng(seed + SEED_OFFSET)
    t0 = time.time()
    for r in range(args.n_real):
        noise = sc * colour(rng.normal(0, 1, I0.shape), H)
        Pn = sims.dedisperse_grid(noise, *grid, dt_nu)
        P = P0 + Pn
        Sp = structure_parameter(P, kc)
        out["Sp_d0"][r] = Sp[j0]
        out["Sp_max"][r] = Sp.max()

        # The paper's procedure, with sigma_t^2 estimated from the profile.
        res = dm_uncertainty(P, d, kc, dt_nu)
        out["sig_S_white"][r] = res["sig_S"]
        if res["ok"]:
            out["d_fit_white"][r] = res["d_fit"]
            out["sig_DM_white"][r] = res["sig_DM"]
            out["n_win_white"][r] = res["n_win"]

        # The true variance only, and the true covariance, given to dm_uncertainty as cov_n.
        for model, cov_n in [("diag", cov_diag), ("full", cov_full)]:
            res = dm_uncertainty(P, d, kc, dt_nu, cov_n=cov_n)
            out[f"sig_S_{model}"][r] = res["sig_S"]
            if not res["ok"]:
                continue
            out[f"d_fit_{model}"][r], out[f"sig_DM_{model}"][r] = res["d_fit"], res["sig_DM"]
            win = res["win"]
            out[f"n_win_{model}"][r] = res["n_win"]

            # Without the second-order term: ||a|| + e_par over the same window.
            if model == "full":
                e_par = np.sum(a_hat * sims.structure_vectors(Pn, kc), axis=1)
                fit = fit_parabola(d[win], norm_a[win] + e_par[win])
                if fit is not None:
                    out["d_par_full"][r] = fit[0]

        if (r + 1) % 50 == 0:
            print(f"  {r + 1}/{args.n_real}, {time.time() - t0:.0f} s", flush=True)

    meta = dict(spectrum=spectrum, snr=snr, n_real=args.n_real, seed=seed + SEED_OFFSET, kc=kc,
                d0=p["d0"], sc=sc, rho=RHO, f_knee=F_KNEE, sig_DM_pred=sig_pred,
                sig_DM_pred_white=p["sig_DM"], start=grid[0], step=grid[1], n=grid[2], j0=j0,
                var_per_sample_ratio=float(autocov[0]), e_over_a=float(e_over_a),
                e_over_a_white=float(e_over_a_white), runtime=time.time() - t0)
    os.makedirs(sims.VALIDATION_RESULTS, exist_ok=True)
    path = sims.results_path(f"coloured_{spectrum}_snr{snr}.npz")
    np.savez_compressed(path, meta=json.dumps(meta), **out)
    print(f"wrote {path}", flush=True)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spectrum", choices=["ar1", "red"])
    parser.add_argument("snr", type=int, choices=[15, 40, 200])
    parser.add_argument("--n_real", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    main()
