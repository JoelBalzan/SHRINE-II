"""Monte Carlo check of the noise covariance between trial DMs, V_jl = Cov[e_par(d_j), e_par(d_l)]
(paper Appendix 2, Eqs. covprof, cov; shrine2.covariance.noise_covariance).

A synthetic scattered burst across the CRAFT band (336 x 1 MHz channels, 10-us samples) gives the
coefficients w(t; d_j) at each trial DM. Each realisation draws white noise in every channel of the
dynamic spectrum, dedisperses it at every trial DM exactly as the data are (a Fourier phase ramp per
channel, then a sum over channels), and forms e_par(d_j) = sum_t w(t; d_j) n(t; d_j). The empirical
covariance of e_par over the realisations is compared with noise_covariance, element by element, and
through std[c1], the quantity sigma_DM needs (Eq. varc1).

The noise is white (uncorrelated between samples and between channels). Options add a noise level
that rises during the burst, as self-noise does, unequal noise in the channels, and unevenly spaced
trial DMs.

usage: python mc_covariance.py [--n_real 5000] [--self_noise 0] [--unequal] [--uneven]
                               [--span 0.05] [--n_dm 41] [--kc 28] [--out mc_covariance.png]
"""
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shrine2.covariance import noise_covariance
from shrine2.structure import coefficients

# The CRAFT band of the paper's simulations: 336 channels of 1 MHz from 751.5 to 1087.5 MHz,
# 10-us samples, 6 ms of data, delays measured from the centre of the band.
K_DM = 4.148808  # dispersion constant, ms GHz^2 cm^3 pc^-1
NU = (752.0 + np.arange(336)) / 1e3  # channel centres, GHz
NU_REF = 0.9195  # GHz
DT = 0.01  # sample interval, ms
N = 600  # samples per profile
DT_NU = K_DM * (NU**-2 - NU_REF**-2) / DT  # each channel's delay per unit DM, samples
FR = np.fft.rfftfreq(N)  # Fourier frequency, cycles per sample


def burst_dynamic_spectrum():
    """The noiseless burst, already dedispersed at its true DM: a Gaussian of 0.1 ms (sigma) at
    1.5 ms, scattered by tau = 0.2 ms (nu / 1 GHz)^-4 in each channel.

    Returns:
        I(nu, t), shape (336, N).
    """
    t = np.arange(N) * DT
    pulse = np.exp(-0.5 * ((t - 1.5) / 0.1) ** 2)
    rows = []
    for nu in NU:
        tau = 0.2 * nu**-4
        tail = np.exp(-t / tau)
        tail /= tail.sum()
        # Scattering: the pulse convolved with a one-sided exponential (circular; the burst is far
        # from the end of the window).
        rows.append(np.fft.irfft(np.fft.rfft(pulse) * np.fft.rfft(tail), N))
    return np.array(rows)


def dedisperse(spectra, d):
    """Profiles at each trial DM from the Fourier transforms of the channels: shift channel nu by
    d dt_nu samples (a phase ramp) and sum over channels, as the data are dedispersed.

    Args:
        spectra: rfft of each channel's time series, shape (336, N // 2 + 1).
        d: trial DMs relative to the dynamic spectrum's DM, shape (n_dm,).
    Returns:
        profiles, shape (n_dm, N).
    """
    rows = []
    for dj in d:
        ramp = np.exp(2j * np.pi * FR[None, :] * (dj * DT_NU)[:, None])
        rows.append(np.fft.irfft((spectra * ramp).sum(axis=0), N))
    return np.array(rows)


def main():
    """Run the Monte Carlo, print the comparison and plot it."""
    args = get_args()
    rng = np.random.default_rng(args.seed)

    # The trial DMs, relative to the burst's true DM; d_ref, where sigma_t^2 is set, is the middle.
    if args.uneven:
        d = np.sort(rng.uniform(-args.span, args.span, args.n_dm))
    else:
        d = np.linspace(-args.span, args.span, args.n_dm)
    j_ref = args.n_dm // 2
    d_ref = d[j_ref]

    # The coefficients w(t; d_j) of the noiseless burst at each trial DM (Eq. ahat).
    I0 = burst_dynamic_spectrum()
    W = coefficients(dedisperse(np.fft.rfft(I0, axis=1), d), args.kc)

    # sigma_t^2, the profile-noise variance at d_ref: constant, or rising by a factor
    # 1 + self_noise at the peak of the burst; and each channel's share of it.
    envelope = dedisperse(np.fft.rfft(I0, axis=1), np.array([d_ref]))[0]
    sig2 = 1 + args.self_noise * envelope / envelope.max()
    if args.unequal:
        share = rng.uniform(0.5, 1.5, NU.size)
    else:
        share = np.ones(NU.size)
    share = share / share.sum()

    # The Monte Carlo: white noise in each channel of the dynamic spectrum at d_ref, with variance
    # share_nu sigma_t^2, dedispersed at every trial DM; e_par(d_j) = sum_t w(t; d_j) n(t; d_j).
    # The phase ramps of dedisperse(), for every trial DM and channel, are computed once.
    channel_sd = np.sqrt(share[:, None] * sig2[None, :])
    ramps = np.exp(2j * np.pi * FR[None, None, :] * ((d - d_ref)[:, None] * DT_NU[None, :])[:, :, None])
    e = np.empty((args.n_real, d.size))
    for r in range(args.n_real):
        noise = rng.normal(size=(NU.size, N)) * channel_sd
        # Each channel shifted by its delay at each trial DM and summed over channels.
        summed = np.einsum("jcf,cf->jf", ramps, np.fft.rfft(noise, axis=1))
        profiles = np.fft.irfft(summed, N, axis=1)
        e[r] = np.sum(W * profiles, axis=1)
    V_mc = np.cov(e.T)

    # The prediction (Appendix 2).
    V = noise_covariance(W, sig2, np.outer(d - d_ref, DT_NU), share)

    # Element by element: the largest difference relative to the largest element, and the
    # correlation coefficients.
    sd = np.sqrt(np.diag(V))
    sd_mc = np.sqrt(np.diag(V_mc))
    rho = V / np.outer(sd, sd)
    rho_mc = V_mc / np.outer(sd_mc, sd_mc)
    # Sampling error of each correlation coefficient, (1 - rho^2) / sqrt(n_real).
    rho_err = np.clip(1 - rho**2, 0, None) / np.sqrt(args.n_real)

    # std[c1] for a parabola fitted over all the trial DMs, about d_ref (Eqs. lsq, varc1), and its
    # Monte Carlo error.
    A = np.vstack([np.ones(d.size), d - d_ref, (d - d_ref) ** 2]).T
    b1 = np.linalg.pinv(A)[1]
    c1 = e @ b1
    ratio_c1 = np.sqrt(b1 @ V @ b1) / c1.std()
    ratio_c1_err = 1 / np.sqrt(2 * args.n_real)

    print(f"{args.n_real} realisations, {d.size} trial DMs over {d[0]:+.3f} to {d[-1]:+.3f} pc cm^-3"
          f"{' (uneven)' if args.uneven else ''}, k_c = {args.kc}, self-noise factor "
          f"{1 + args.self_noise:g} at the peak, {'unequal' if args.unequal else 'equal'} channels")
    print(f"  max |V - V_mc| / max V = {np.abs(V - V_mc).max() / V.max():.4f}")
    print(f"  diagonal: V / V_mc = {np.min(sd**2 / sd_mc**2):.3f} to {np.max(sd**2 / sd_mc**2):.3f}"
          f" (sampling error {np.sqrt(2 / args.n_real):.3f})")
    print(f"  correlation: max |rho - rho_mc| / its sampling error = "
          f"{np.max(np.abs(rho - rho_mc) / np.maximum(rho_err, 1e-12)):.2f}; rho between the "
          f"end trial DMs {rho[0, -1]:.4f} (MC {rho_mc[0, -1]:.4f})")
    print(f"  std[c1] predicted / Monte Carlo = {ratio_c1:.3f} +- {ratio_c1_err:.3f}")

    plot(d, d_ref, j_ref, V, V_mc, rho, rho_mc, rho_err, ratio_c1, ratio_c1_err, args)


def plot(d, d_ref, j_ref, V, V_mc, rho, rho_mc, rho_err, ratio_c1, ratio_c1_err, args):
    """(a) The correlation of e_par(d_j) with e_par(d_ref) against d_j - d_ref, the shape of V.
    (b) The standard deviation of e_par(d_j), sqrt(V_jj), the level of V. Monte Carlo and prediction.
    The Monte Carlo points share the same noise realisations, so their errors are correlated.

    Args:
        d, d_ref, j_ref: the trial DMs, the reference one and its index.
        V, V_mc: predicted and Monte Carlo covariance.
        rho, rho_mc, rho_err: the correlation coefficients and the sampling error of the MC ones.
        ratio_c1, ratio_c1_err: std[c1] predicted / Monte Carlo, and its error.
        args: the parsed flags.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8, 3.4))
    x = d - d_ref
    ax_a.plot(x, rho[j_ref], color="k", lw=1.2, label="Appendix 2")
    ax_a.errorbar(x, rho_mc[j_ref], yerr=rho_err[j_ref], fmt="o", ms=3, color="#dc267f",
                  label=f"Monte Carlo ({args.n_real} realisations)")
    ax_a.set_xlabel(r"$d_j-d_{\rm ref}$ [pc cm$^{-3}$]")
    ax_a.set_ylabel(r"Corr$[e_\parallel(d_j),\,e_\parallel(d_{\rm ref})]$")
    ax_a.legend(fontsize=8, frameon=False, loc="lower center")
    ax_a.set_title("(a) correlation with the reference trial DM", fontsize=9)

    # sqrt(V_jj) in units of sigma_S at d_ref, the predicted sqrt(V_jj) there (Eq. sigS); the
    # sampling error of a standard deviation is sd / sqrt(2 n_real).
    scale = np.sqrt(V[j_ref, j_ref])
    sd = np.sqrt(np.diag(V)) / scale
    sd_mc = np.sqrt(np.diag(V_mc)) / scale
    ax_b.plot(x, sd, color="k", lw=1.2)
    ax_b.errorbar(x, sd_mc, yerr=sd_mc / np.sqrt(2 * args.n_real), fmt="o", ms=3, color="#dc267f")
    ax_b.set_xlabel(r"$d_j-d_{\rm ref}$ [pc cm$^{-3}$]")
    ax_b.set_ylabel(r"std$[e_\parallel(d_j)]\,/\,\sigma_{\mathcal{S}}(d_{\rm ref})$")
    ax_b.set_title(f"(b) standard deviation; std$[c_1]$ predicted/MC = {ratio_c1:.3f} $\\pm$ {ratio_c1_err:.3f}",
                   fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"  plot: {args.out}")


def get_args():
    """Parse the command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    parser = ArgumentParser(description="Monte Carlo check of V_jl (paper Appendix 2).",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n_real", type=int, default=5000, help="noise realisations")
    parser.add_argument("--n_dm", type=int, default=41, help="number of trial DMs")
    parser.add_argument("--span", type=float, default=0.05,
                        help="trial DMs from -span to +span about the true DM, pc cm^-3")
    parser.add_argument("--uneven", action="store_true", help="unevenly spaced trial DMs")
    parser.add_argument("--self_noise", type=float, default=0.0,
                        help="noise variance rises by 1 + this at the peak of the burst")
    parser.add_argument("--unequal", action="store_true", help="unequal noise in the channels")
    parser.add_argument("--kc", type=int, default=28, help="filter cutoff k_c")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--out", default="mc_covariance.png", help="plot file")
    return parser.parse_args()


if __name__ == "__main__":
    main()
