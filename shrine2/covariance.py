"""The covariance of the noise between trial DMs, V_jl = Cov[e_par(d_j), e_par(d_l)].

Paper: Appendix 2 (Eqs. epchan, cov).

Dedispersing at another trial DM only shifts the channels relative to one another, by
(d_j - d_l) dt_nu samples for channel nu, so the same channel noise appears in both profiles.
With s_nu each channel's share of the profile-noise variance (s_nu = 1/N_nu for equal channels,
as in Eq. cov):

    V_jl = 1/2 sum_nu s_nu sum_t (sigma_t^2 + sigma_t'^2) w(t; d_j) w(t'; d_l),
    t' = t + (d_j - d_l) dt_nu.

The lags are not whole samples, so V is evaluated in the Fourier domain, where a lag is a phase
ramp:

    sum_t a(t) b(t + lag) = (1/M) sum_f conj(A_f) B_f exp(2 pi i f lag),

with both series zero-padded to M = 2N samples so that the lags do not wrap round.
The trial DMs must be uniformly spaced: then V_jl depends on the lag only through m = j - l.
"""
import numpy as np


def noise_covariance(W, sig2, dt_nu, step, share=None):
    """The covariance of the noise in Sp between each pair of trial DMs in the fit window,
    V_jl = Cov[e_par(d_j), e_par(d_l)] (Appendix 2, Eq. cov).

    The n trial DMs must be consecutive and uniformly spaced.

    Args:
        W: w(t; d_j) at each trial DM in the window, shape (n, N), one row per trial DM, in order
            of d_j.
        sig2: sigma_t^2, shape (N,).
        dt_nu: each channel's dispersion delay per unit DM, in samples, shape (n_chan,).
        step: trial-DM spacing d_(j+1) - d_j, pc cm^-3.
        share: each channel's share of the noise variance, shape (n_chan,); None for equal shares.
    Returns:
        V, shape (n, n), in the units of Sp squared.
    """
    n, N = W.shape

    # Zero-pad to M = 2N samples so that the lags do not wrap round.
    M = 2 * N
    f = np.fft.rfftfreq(M)                               # Fourier frequency, cycles per sample

    # rfft keeps f >= 0 only. Each f > 0 stands for +f and -f, so it counts twice in the sum over
    # all frequencies; f = 0 and the Nyquist frequency (M is even) count once.
    n_f = np.full(f.size, 2.0)
    n_f[0] = 1.0
    n_f[-1] = 1.0

    # Each channel's share s_nu of the noise variance, normalised to sum to 1.
    if share is None:
        s = np.ones(dt_nu.size)
    else:
        s = np.asarray(share, float)
    s = s / s.sum()

    # Transforms of sigma_t^2 w(t; d_j) and of w(t; d_j), one row per trial DM.
    F_sw = np.fft.rfft(sig2 * W, M, axis=1)
    F_w = np.fft.rfft(W, M, axis=1)

    V = np.empty((n, n))
    for m in range(n):
        # Every pair of trial DMs m steps apart has the same lag in each channel, in samples.
        lag = m * step * dt_nu

        # Each channel's lag as a phase ramp over the Fourier frequencies, averaged over channels.
        phase_ramps = np.exp(2j * np.pi * lag[:, None] * f[None, :])
        ramp = s @ phase_ramps

        # The pairs (j, l) with j - l = m.
        j = np.arange(m, n)
        l = j - m

        # Cross-spectra with sigma_t^2 taken at t (the sample's place at d_j) and with sigma_t'^2
        # taken at t' (its place at d_l).
        at_t = np.conj(F_sw[j]) * F_w[l]
        at_tp = np.conj(F_w[j]) * F_sw[l]

        # Average the two, apply the lag and sum over all Fourier frequencies (1/M normalisation).
        v = 0.5 * np.real((at_t + at_tp) * ramp * n_f).sum(axis=1) / M
        V[j, l] = v
        V[l, j] = v
    return V
