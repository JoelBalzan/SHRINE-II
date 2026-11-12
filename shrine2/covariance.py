"""The covariance of the noise between trial DMs, V_jl = Cov[e_par(d_j), e_par(d_l)].

Paper: Appendix 2 (Eqs. epchan, backproj, cov).

Dedispersing at another trial DM does not change the noise in any channel; it only shifts the
channels relative to one another before they are summed. So the noise at every trial DM can be
written in terms of one set of noise samples: n_nu(t), the noise in channel nu of the dynamic
spectrum dedispersed at d_ref, the trial DM at which sigma_t^2 is measured (d_max). Dedispersing
at d_j instead shifts channel nu by a further

    delta_j_nu = (d_j - d_ref) dt_nu  samples,

so e_par(d_j) = sum_t w(t; d_j) sum_nu n_nu(t + delta_j_nu). Relabelling t in each channel,

    e_par(d_j) = sum_nu sum_t g_j(nu, t) n_nu(t),    g_j(nu, t) = w(t - delta_j_nu; d_j)
                                                                          (Eqs. epchan, backproj)

g_j is w at d_j copied into every channel and shifted by delta_j_nu: the dedispersion to d_j run
in reverse. It is the weight of each dynamic-spectrum noise sample in e_par(d_j).

If the noise is uncorrelated between samples and between channels, and channel nu carries a share
s_nu of sigma_t^2, only products of the same sample contribute, each with that sample's variance:

    V_jl = sum_nu s_nu sum_t sigma_t^2 g_j(nu, t) g_l(nu, t)    (Eq. cov),

the dot product of the two weight maps, weighted by the noise variance of each sample.

The shifts are fractional, so g_j is made with the same Fourier phase ramp as the dedispersion,
with the opposite sign. The trial DMs need not be uniformly spaced.
"""
import numpy as np


def noise_covariance(W, sig2, d, d_ref, dt_nu, share=None):
    """The covariance of the noise in Sp between each pair of trial DMs in the fit window,
    V_jl = Cov[e_par(d_j), e_par(d_l)] (Appendix 2, Eq. cov).

    Args:
        W: w(t; d_j) at each trial DM in the window, shape (n, N), one row per trial DM.
        sig2: sigma_t^2, the profile-noise variance measured at d_ref, shape (N,).
        d: the trial DMs of the rows of W, shape (n,), pc cm^-3.
        d_ref: the trial DM at which sig2 was measured (d_max), pc cm^-3.
        dt_nu: each channel's dispersion delay per unit DM, in samples, shape (n_chan,).
        share: each channel's share of the noise variance, shape (n_chan,); None for equal shares.
    Returns:
        V, shape (n, n), in the units of Sp squared.
    """
    n, N = W.shape  # n trial DMs, N samples per profile
    d = np.asarray(d, float)
    dt_nu = np.asarray(dt_nu, float)

    # Each channel's share s_nu of the noise variance, normalised to sum to 1.
    if share is None:
        s = np.ones(dt_nu.size)
    else:
        s = np.asarray(share, float)
    s = s / s.sum()

    # delta_j_nu: how far dedispersing at d_j rather than d_ref shifts channel nu, in samples.
    delay = np.outer(d - d_ref, dt_nu)  # shape (n, n_chan)

    # Zero-pad w on both sides, so that the shifted copies do not wrap round: by N/2, or by more
    # if a shift is longer. The noise variance outside the profile is taken as its value at the
    # nearest end.
    pad = max(N // 2, int(np.ceil(np.abs(delay).max())) + 1)
    M = N + 2 * pad
    W_padded = np.zeros((n, M))
    W_padded[:, pad:pad + N] = W
    sigma = np.sqrt(np.pad(sig2, pad, mode="edge"))

    # w(t; d_j) in the Fourier domain, where a shift is a phase ramp.
    f = np.fft.rfftfreq(M)  # Fourier frequency, cycles per sample
    F_w = np.fft.rfft(W_padded, axis=1)

    V = np.zeros((n, n))
    for nu in range(dt_nu.size):
        # g_j(nu, t) = w(t - delta_j_nu; d_j) for every trial DM j (Eq. backproj): a shift of
        # delta_j_nu samples later, the phase ramp exp(-2 pi i f delta_j_nu).
        ramp = np.exp(-2j * np.pi * np.outer(delay[:, nu], f))
        G = np.fft.irfft(F_w * ramp, M, axis=1)

        # Times the noise standard deviation of each sample in this channel, sqrt(s_nu sigma_t^2),
        # so that G @ G.T is this channel's term of Eq. cov.
        G = G * (np.sqrt(s[nu]) * sigma)
        V += G @ G.T
    return V
