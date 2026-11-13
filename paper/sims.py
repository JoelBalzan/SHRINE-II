"""Shared pieces of the paper's simulations (Balzan et al., in prep.).

The noiseless FIRES bursts, their channel delays, dedispersion, the noise level for a given S/N,
the pilot that sets k_c, d_0 and the trial DMs, and S23's uncertainties, for comparison.

The structure-maximising DM and sigma_DM always come from shrine2: dm_uncertainty for the noisy
realisations, and its building blocks (fit_window, fit_parabola, coefficients, noise_covariance)
for the noiseless predictions and for fits over a given window. Every number in the paper is
therefore computed by the published code.

Conventions, as in the paper's Section 2.1:
    - each burst is a dynamic spectrum I(nu, t) dedispersed at its true DM, so a trial DM d is an
      offset from the true DM;
    - dedispersing at d moves channel nu earlier by d dt_nu samples (a Fourier phase ramp, so
      fractional shifts are exact), with dt_nu measured from 919.5 MHz, near the band centre;
    - the noise is white, Gaussian and equal in every channel, at the level FIRES uses for a given
      S/N over the window that holds 95% of the fluence.
"""
import os
import sys

import numpy as np

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))

# Use shrine2 from this checkout, whether or not it is installed.
sys.path.insert(0, os.path.dirname(PAPER_DIR))
from shrine2.covariance import noise_covariance  # noqa: E402
from shrine2.fit import fit_parabola, fit_window  # noqa: E402
from shrine2.noise import sigma_S  # noqa: E402
from shrine2.structure import (  # noqa: E402
    choose_kc,
    coefficients,
    forward_difference_matrix,
    smooth,
    structure_parameter,
)

BURST_DIR = os.path.join(PAPER_DIR, "bursts")
VALIDATION_RESULTS = os.path.join(PAPER_DIR, "sec4_validation", "results")

K_DM = 4.148808  # dispersion constant, ms GHz^2 cm^3 pc^-1
F_REF = 919.5  # reference frequency of the channel delays, MHz

# The validation bursts and S/N of Section 4, with the number of noise realisations and the seed
# of each case (the seeds of the original runs, so the noise realisations are the same).
SNRS = [15, 25, 40, 70, 120, 200]
CASES = {
    ("single", 15): (500, 302), ("single", 25): (500, 312), ("single", 40): (1000, 301),
    ("single", 70): (500, 313), ("single", 120): (500, 303), ("single", 200): (500, 314),
    ("double", 15): (500, 315), ("double", 25): (500, 304), ("double", 40): (1000, 311),
    ("double", 70): (500, 305), ("double", 120): (500, 316), ("double", 200): (500, 306),
}

# The noise realisation of the example burst (one-component burst, S/N = 40) of Figures 1, 3, 4,
# 6 and 8.
EXAMPLE_SEED = 111


# ------------------------------------------------------------------------------------- bursts
def load_burst(tag):
    """One noiseless FIRES burst from bursts/<tag>.npz.

    Args:
        tag: "single", "double" (Section 2.1) or "drift000" ... "drift080" (Section 5.1).
    Returns:
        I: the dynamic spectrum I(nu, t) at the true DM, shape (n_chan, N), channels from the top
            of the band down.
        freq: channel centres, MHz, shape (n_chan,).
        time: sample times, ms, shape (N,).
    """
    data = np.load(os.path.join(BURST_DIR, f"{tag}.npz"))
    return data["I"], data["freq"], data["time"]


def channel_delays(freq, dt, f_ref=F_REF):
    """Each channel's dispersion delay per unit DM relative to f_ref, in samples.

    (shrine2.profiles.channel_delays gives the same for CRAFT voltages, from the band's centre
    frequency and bandwidth; here the channels come from FIRES.)

    Args:
        freq: channel centres, MHz.
        dt: sample interval, ms.
        f_ref: reference frequency, MHz.
    Returns:
        dt_nu, shape (n_chan,), samples per pc cm^-3; positive below f_ref.
    """
    return K_DM * ((freq / 1e3) ** -2 - (f_ref / 1e3) ** -2) / dt


def one_sample_dm_step(dt_nu):
    """The DM step that delays the bottom of the band by one sample relative to the top.

    Args:
        dt_nu: channel delays per unit DM, samples.
    Returns:
        the step, pc cm^-3.
    """
    return 1.0 / (dt_nu.max() - dt_nu.min())


# -------------------------------------------------------------------------------------- noise
def boxcar_window(profile, frac=0.95):
    """The narrowest window holding a fraction frac of the profile's fluence (FIRES,
    fires.utils.profiles.boxcar_width, copied so that FIRES is not needed here).

    Args:
        profile: frequency-summed profile, shape (N,).
        frac: fraction of the fluence.
    Returns:
        (first, last): the window's first and last sample.
    """
    prof = np.nan_to_num(profile)
    n = prof.size
    target = frac * prof.sum()
    cumsum = np.cumsum(prof)
    best_width = n
    best = (0, n - 1)
    for start in range(n):
        if start > 0:
            before = cumsum[start - 1]
        else:
            before = 0.0
        ends = np.flatnonzero(cumsum >= before + target)
        if ends.size > 0:
            width = ends[0] - start + 1
            if width < best_width:
                best_width = width
                best = (start, ends[0])
    return best


def channel_noise_sigma(I0, snr):
    """The noise standard deviation per channel and sample that gives the burst a boxcar S/N of
    snr (FIRES's compute_required_sefd).

    With equal noise sigma_ch in each of the n_chan channels, the frequency-summed profile has
    noise sqrt(n_chan) sigma_ch per sample, and the boxcar of n_on samples holding 95% of the
    fluence F_on has S/N = F_on / (sqrt(n_chan n_on) sigma_ch).

    Args:
        I0: noiseless dynamic spectrum, shape (n_chan, N).
        snr: the S/N.
    Returns:
        sigma_ch.
    """
    prof = I0.sum(axis=0)
    first, last = boxcar_window(prof)
    n_on = last - first + 1
    fluence = prof[first:last + 1].sum()
    return fluence / (snr * np.sqrt(I0.shape[0] * n_on))


# ------------------------------------------------------------------------------- dedispersion
def dedisperse(x, d, dt_nu):
    """The profiles I(d_j, t) at the trial DMs d: each channel moved earlier by d_j dt_nu samples
    (a Fourier phase ramp, circular in time), then summed over channels.

    Args:
        x: dynamic spectrum, shape (n_chan, N), at the true DM.
        d: trial DMs (offsets from the true DM), shape (n_d,).
        dt_nu: channel delays per unit DM, samples, shape (n_chan,).
    Returns:
        I(d_j, t), shape (n_d, N).
    """
    N = x.shape[1]
    f = np.fft.rfftfreq(N)
    X = np.fft.rfft(x, axis=1)
    rows = []
    for dj in np.atleast_1d(d):
        ramp = np.exp(2j * np.pi * f[None, :] * (dj * dt_nu)[:, None])
        rows.append(np.fft.irfft((X * ramp).sum(axis=0), N))
    return np.array(rows)


def dedisperse_grid(x, start, step, n, dt_nu):
    """dedisperse() at the n uniformly spaced trial DMs start + j step, j = 0 ... n - 1.

    The same result, faster: the phase ramp of trial DM j + 1 is that of trial DM j times the ramp
    of one step, so each trial DM costs one multiplication instead of an exponential.

    Args:
        x: dynamic spectrum, shape (n_chan, N).
        start, step: the first trial DM and the step, pc cm^-3.
        n: number of trial DMs.
        dt_nu: channel delays per unit DM, samples, shape (n_chan,).
    Returns:
        I(d_j, t), shape (n, N).
    """
    N = x.shape[1]
    f = np.fft.rfftfreq(N)
    A = np.fft.rfft(x, axis=1) * np.exp(2j * np.pi * f[None, :] * (start * dt_nu)[:, None])
    one_step = np.exp(2j * np.pi * f[None, :] * (step * dt_nu)[:, None])
    Z = np.empty((n, f.size), complex)
    for j in range(n):
        Z[j] = A.sum(axis=0)
        A *= one_step
    return np.fft.irfft(Z, N, axis=1)


def dedisperse_spectrum(x, d, dt_nu):
    """The dynamic spectrum itself dedispersed at one trial DM d (for the figures).

    Args:
        x: dynamic spectrum, shape (n_chan, N), at the true DM.
        d: the trial DM.
        dt_nu: channel delays per unit DM, samples.
    Returns:
        the dedispersed dynamic spectrum, shape (n_chan, N).
    """
    N = x.shape[1]
    f = np.fft.rfftfreq(N)
    ramp = np.exp(2j * np.pi * f[None, :] * (d * dt_nu)[:, None])
    return np.fft.irfft(np.fft.rfft(x, axis=1) * ramp, N, axis=1)


def coarse_grid(dt_nu):
    """The wide, coarse trial DMs over which k_c is chosen: +-200 one-sample steps, every two.

    Args:
        dt_nu: channel delays per unit DM, samples.
    Returns:
        (start, step, n) for dedisperse_grid.
    """
    d1 = one_sample_dm_step(dt_nu)
    return -200 * d1, 2 * d1, 201


# ------------------------------------------------------------------------------------- fits
def fit_over(P, d, win, j_ref, kc, dt_nu, sig2):
    """The parabola fit and sigma_DM over a given set of trial DMs (Section 3.3.4), with
    shrine2's building blocks, as in shrine2.dm_uncertainty but with the window, the reference
    trial DM and sigma_t^2 given.

    Args:
        P: I(d_j, t) at every trial DM, shape (n_d, N).
        d: the trial DMs, shape (n_d,).
        win: boolean, shape (n_d,): the trial DMs to fit.
        j_ref: the trial DM at which sigma_t^2 holds (d_max in dm_uncertainty).
        kc: filter cutoff.
        dt_nu: channel delays per unit DM, samples.
        sig2: sigma_t^2, shape (N,).
    Returns:
        dict(d_fit, c, B, V, sig_DM, c2_err), or None if the parabola has no maximum.
    """
    Sp = structure_parameter(P[win], kc)
    fit = fit_parabola(d[win], Sp)
    if fit is None:
        return None
    d_fit, c, B = fit

    # V between the trial DMs in the window (Appendix 2), then Eqs. varc1, final and c2err.
    W = coefficients(P[win], kc)
    delay = np.outer(d[win] - d[j_ref], dt_nu)
    V = noise_covariance(W, sig2, delay)
    sig_DM = np.sqrt(B[1] @ V @ B[1]) / abs(2 * c[2])
    c2_err = np.sqrt(B[2] @ V @ B[2]) / abs(c[2])
    return dict(d_fit=d_fit, c=c, B=B, V=V, sig_DM=sig_DM, c2_err=c2_err)


def noiseless_fit(P0, d, kc, dt_nu, sig2):
    """What Section 3.3 gives for the noiseless burst, with the true sigma_t^2: the window of
    trial DMs within sigma_S of the peak of ||a(d)||, the vertex of the parabola fitted over it
    (this is d_0) and the predicted sigma_DM.

    Args:
        P0: noiseless I(d_j, t), shape (n_d, N).
        d: the trial DMs, shape (n_d,).
        kc: filter cutoff.
        dt_nu: channel delays per unit DM, samples.
        sig2: the true sigma_t^2, shape (N,).
    Returns:
        dict(d0, sig_DM, sig_S, win, j_max, c2, c2_err); raises RuntimeError if the window reaches
        the end of the trial DMs or the parabola has no maximum.
    """
    norm_a = structure_parameter(P0, kc)
    j_max = int(np.argmax(norm_a))
    sig_S_true = sigma_S(coefficients(P0[j_max], kc), sig2)
    win = fit_window(norm_a, sig_S_true, j_max)
    if win is None:
        raise RuntimeError("the noiseless fit window reaches the end of the trial DMs")
    fit = fit_over(P0, d, win, j_max, kc, dt_nu, sig2)
    if fit is None:
        raise RuntimeError("the noiseless parabola has no maximum")
    return dict(d0=fit["d_fit"], sig_DM=fit["sig_DM"], sig_S=sig_S_true, win=win, j_max=j_max,
                c2=fit["c"][2], c2_err=fit["c2_err"])


def validation_grid(d0, sig_DM, d1):
    """The trial DMs of the Monte Carlo runs (Section 4): steps of sigma_DM/6 out to
    max(15 sigma_DM, 20 one-sample steps) either side of d_0, with d_0 on the grid.

    Args:
        d0: structure-maximising DM of the noiseless burst.
        sig_DM: its predicted sigma_DM.
        d1: one-sample DM step.
    Returns:
        (start, step, n) for dedisperse_grid.
    """
    step = sig_DM / 6
    half_width = max(15 * sig_DM, 20 * d1)
    n = 2 * int(np.ceil(half_width / step)) + 1
    start = d0 - step * (n // 2)
    return start, step, n


def pilot(I0, dt_nu, sc, rng, n_kc=5):
    """k_c, d_0, the predicted sigma_DM and the trial DMs for one burst and S/N (Section 4).

    k_c is the median of S23's k_c (without the extended search) over n_kc noise realisations on
    the coarse trial DMs, then held fixed. d_0 and sigma_DM are noiseless_fit's, iterated: the
    trial DMs are spaced by sigma_DM/6, so they are refined until sigma_DM changes by less than
    1%.

    Args:
        I0: noiseless dynamic spectrum, shape (n_chan, N).
        dt_nu: channel delays per unit DM, samples.
        sc: noise standard deviation per channel and sample.
        rng: numpy Generator; draws n_kc noise realisations (the Monte Carlo continues with it).
        n_kc: number of realisations for k_c.
    Returns:
        dict(kc, d0, sig_DM, sig_S, start, step, n, d1, sig2), with sig2 the true sigma_t^2 of the
        profile (n_chan sc^2 at every sample).
    """
    n_chan, N = I0.shape
    d1 = one_sample_dm_step(dt_nu)
    start, step, n = coarse_grid(dt_nu)

    # k_c: the median of S23's procedure over n_kc noise realisations.
    kcs = []
    for _ in range(n_kc):
        noisy = I0 + rng.normal(0, sc, I0.shape)
        kcs.append(choose_kc(dedisperse_grid(noisy, start, step, n, dt_nu), extend=False))
    kc = int(np.median(kcs))

    # The true sigma_t^2 of the frequency-summed profile: n_chan channels of variance sc^2.
    sig2 = np.full(N, n_chan * sc**2)

    # A first guess: the highest coarse trial DM, with trial DMs one sample step apart.
    norm_a = structure_parameter(dedisperse_grid(I0, start, step, n, dt_nu), kc)
    d_peak = start + step * np.argmax(norm_a)
    grid = (d_peak - 60 * d1, d1 / 2, 241)

    # Refine: fit on the current trial DMs, then space them by the new sigma_DM / 6.
    sig_DM_old = np.inf
    for _ in range(10):
        d = grid[0] + grid[1] * np.arange(grid[2])
        pred = noiseless_fit(dedisperse_grid(I0, *grid, dt_nu), d, kc, dt_nu, sig2)
        grid = validation_grid(pred["d0"], pred["sig_DM"], d1)
        if abs(pred["sig_DM"] / sig_DM_old - 1) < 0.01:
            break
        sig_DM_old = pred["sig_DM"]

    return dict(kc=kc, d0=pred["d0"], sig_DM=pred["sig_DM"], sig_S=pred["sig_S"],
                start=grid[0], step=grid[1], n=grid[2], d1=d1, sig2=sig2)


def crossings(Sp, d, j_max, level):
    """Where Sp(d) falls through level either side of its peak, interpolated linearly between
    trial DMs: the ends of the sigma_S crossing range (Section 3.1) when level = Sp_max - sigma_S.

    Args:
        Sp: Sp(d_j), shape (n_d,).
        d: trial DMs (uniform), shape (n_d,).
        j_max: index of the peak.
        level: the level.
    Returns:
        (d_low, d_high), or None if Sp does not fall through level on both sides.
    """
    lo = j_max
    while lo > 0 and Sp[lo - 1] >= level:
        lo -= 1
    hi = j_max
    while hi < Sp.size - 1 and Sp[hi + 1] >= level:
        hi += 1
    if lo == 0 or hi == Sp.size - 1:
        return None
    step = d[1] - d[0]
    d_low = d[lo - 1] + step * (level - Sp[lo - 1]) / (Sp[lo] - Sp[lo - 1])
    d_high = d[hi] + step * (Sp[hi] - level) / (Sp[hi] - Sp[hi + 1])
    return d_low, d_high


def structure_vectors(P, kc):
    """D1 S x for each profile (row), as in shrine2.structure.structure_parameter.

    Args:
        P: profiles, shape (n_d, N).
        kc: filter cutoff.
    Returns:
        D1 S x, shape (n_d, N - 1).
    """
    D1 = forward_difference_matrix(P.shape[-1])
    return smooth(P, kc) @ D1.T


# --------------------------------------------------------------------------------------- S23
def s23_relative_uncertainty(x, kc):
    """S23's relative uncertainty of Sp (Eq. s23rel): ||D1 S Delta_x|| / ||D1 S S x||, with
    Delta_x = x - S x the residual of the smoothing.

    Args:
        x: one profile, shape (N,).
        kc: filter cutoff.
    Returns:
        the relative uncertainty, a float.
    """
    Sx = smooth(x, kc)
    D1 = forward_difference_matrix(x.size)
    residual = D1 @ smooth(x - Sx, kc)
    signal = D1 @ smooth(Sx, kc)
    return np.linalg.norm(residual) / np.linalg.norm(signal)


def s23_interval(P, d, kc):
    """S23's DM interval (Eq. s23int), as in SHRINE: the trial DMs d that pass
    Sp(d) (1 + ||D1 S [Delta(d) - Delta(d_max)]|| / ||D1 S S x(d)||) >= Sp(d_max), about the
    highest trial DM d_max.

    The interval runs from the first trial DM that passes to the first that fails above the last
    that passes, as in SHRINE.

    Args:
        P: I(d_j, t), shape (n_d, N).
        d: trial DMs, shape (n_d,).
        kc: filter cutoff.
    Returns:
        (d_low, d_high, at_edge): the interval's ends, and whether it reaches either end of d
        (then it is a lower limit).
    """
    Sp = structure_parameter(P, kc)
    j_max = int(np.argmax(Sp))
    SP = smooth(P, kc)
    residual = smooth(P - SP, kc)
    D1 = forward_difference_matrix(P.shape[-1])
    change = np.linalg.norm((residual - residual[j_max]) @ D1.T, axis=1)
    signal = np.linalg.norm(smooth(SP, kc) @ D1.T, axis=1)
    passes = Sp * (1 + change / signal) >= Sp[j_max]
    lo = int(np.argmax(passes))
    hi = passes.size - 1 - int(np.argmax(passes[::-1]))
    at_edge = lo == 0 or hi == passes.size - 1
    return d[lo], d[min(hi + 1, d.size - 1)], at_edge


def example():
    """The example burst used throughout the paper (Figures 1, 3, 4, 6 and 8): one noise realisation
    (seed EXAMPLE_SEED) of the one-component burst at S/N = 40, with the pilot of that case
    (k_c, d_0, the predicted sigma_DM) and trial DMs spaced by sigma_DM/6 out to +-0.15 pc cm^-3
    about d_0 (wide enough for S23's interval).

    Returns:
        dict(I0, freq, time, dt_nu, sc, pilot, noise, d, P0, P): the noiseless dynamic spectrum, its
        channels and samples, the channel delays, the noise level per channel, the pilot (as
        sims.pilot), the noise realisation (n_chan, N), the trial DMs, and the noiseless and noisy
        profiles at them.
    """
    I0, freq, time = load_burst("single")
    dt_nu = channel_delays(freq, time[1] - time[0])
    sc = channel_noise_sigma(I0, 40)
    _, seed = CASES[("single", 40)]
    p = pilot(I0, dt_nu, sc, np.random.default_rng(seed))

    step = p["sig_DM"] / 6
    n_half = int(np.ceil(0.15 / step))
    start = p["d0"] - n_half * step
    d = start + step * np.arange(2 * n_half + 1)

    noise = np.random.default_rng(EXAMPLE_SEED).normal(0, sc, I0.shape)
    P0 = dedisperse_grid(I0, start, step, d.size, dt_nu)
    P = P0 + dedisperse_grid(noise, start, step, d.size, dt_nu)
    return dict(I0=I0, freq=freq, time=time, dt_nu=dt_nu, sc=sc, pilot=p, noise=noise, d=d,
                P0=P0, P=P)


def results_path(name):
    """Path of a file in sec4_validation/results (the Monte Carlo results other sections read).

    Args:
        name: file name.
    Returns:
        the path.
    """
    return os.path.join(VALIDATION_RESULTS, name)
