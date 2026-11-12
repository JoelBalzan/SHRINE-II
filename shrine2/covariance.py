"""Covariance of the structure-parameter noise between trial DMs (Appendix 2).

    V_jl = w(d_j) K_jl w(d_l)^T,                            (Eq. covprof)
    K_jl = sum_nu s_nu T_jnu Cov(n) T_lnu^T.                (Eq. cov)

Substituting the second into the first,

    V_jl = sum_nu s_nu [w(d_j) T_jnu] Cov(n) [w(d_l) T_lnu]^T,

so we never form K_jl or T_jnu. w(d_j) T_jnu is w(d_j) shifted by delta_jnu, computed
with the same Fourier phase ramp as the dedispersion. The sum over channels is done one channel
at a time, so that only the shifted rows of one channel, shape (n, N), are held at once.
"""
import numpy as np


def shift_rows(w, delay):
    """w T_jnu: w shifted later by delay samples.

    T_jnu moves channel nu of the noise earlier by delta_jnu, so a row vector multiplied by
    it on the right moves later by the same amount: element t of w T_jnu is w(t - delta_jnu).
    Samples shifted in from outside the profile are zero, as for T_jnu.

    Works on one row and many delays (w of shape (N,), delay of shape (n_chan,)), or on many
    rows with one delay each (w of shape (n, N), delay of shape (n,)).

    Args:
        w: one row w(d_j), shape (N,), or several, shape (n, N).
        delay: the shift of each output row, in samples.
    Returns:
        the shifted rows, shape (len(delay), N).
    """
    N = w.shape[-1]
    M = int(2 ** np.ceil(np.log2(N + np.abs(delay).max() + 1)))  # zero-pad so nothing wraps
    f = np.fft.rfftfreq(M)                                       # cycles per sample
    ramp = np.exp(-2j * np.pi * delay[:, None] * f)              # delay by delta_jnu
    return np.fft.irfft(np.fft.rfft(w, M, axis=-1) * ramp, M, axis=-1)[:, :N]


def noise_covariance(W, cov_n, delay, share=None):
    """V_jl for the trial DMs in the fit window (Eqs. covprof and cov).

    Args:
        W: w(d_j) at each trial DM, shape (n, N), one row per trial DM.
        cov_n: Cov(n), the covariance of the profile noise at d_max. Either its diagonal,
            sigma_t^2, shape (N,), for white noise, or the full matrix, shape (N, N).
        delay: delta_jnu = (d_j - d_max) * dt_nu, in samples, shape (n, n_chan).
        share: s_nu, each channel's share of the profile-noise variance, shape (n_chan,);
            None for equal shares.
    Returns:
        V, shape (n, n).
    """
    n, N = W.shape  # n trial DMs, N samples per profile
    n_chan = delay.shape[1]
    if share is None:
        s = np.full(n_chan, 1.0 / n_chan)
    else:
        s = np.asarray(share, float)
    s = s / s.sum()
    cov_n = np.asarray(cov_n, float)

    V = np.zeros((n, n))
    for nu in range(n_chan):
        # Q[j] = w(d_j) T_jnu: row j shifted by delta_jnu, for every trial DM j.
        Q = shift_rows(W, delay[:, nu])

        # Q Cov(n). White noise: Cov(n) is diagonal, so this scales each sample by sigma_t^2.
        if cov_n.ndim == 1:
            CQ = Q * cov_n
        else:
            CQ = Q @ cov_n

        # This channel's term of V_jl = sum_nu s_nu (Q[j] Cov(n)) . Q[l].
        V += s[nu] * (CQ @ Q.T)
    return V
