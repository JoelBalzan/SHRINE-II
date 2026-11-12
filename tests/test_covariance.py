"""V_jl (Appendix 2, Eqs. covprof, cov) against its definitions."""
import numpy as np

from shrine2.covariance import noise_covariance
from shrine2.noise import sigma_S

rng = np.random.default_rng(2)

# Profiles of N samples summed from n_chan channels, with these delays (samples per unit DM).
N = 128
n_chan = 16
t = np.arange(N)
dt_nu = np.linspace(-20, 20, n_chan)
FR = np.fft.rfftfreq(N)


def smooth_rows(n):
    """Smooth, burst-like coefficient rows w(t; d_j).

    Args:
        n: number of rows (trial DMs).
    Returns:
        rows, shape (n, N): derivatives of a Gaussian that moves slightly from row to row.
    """
    rows = []
    for j in range(n):
        gaussian = np.exp(-0.5 * ((t - 60 - 0.3 * j) / 6) ** 2)
        rows.append(np.gradient(gaussian))
    return np.array(rows)


def test_symmetric_and_diagonal_is_sigma_S_squared():
    """V is symmetric, and at d_ref, where sigma_t^2 is measured, V_jj is sigma_S^2 (Eq. sigS)."""
    W = smooth_rows(7)
    d = 0.05 * np.arange(7)
    sig2 = 1 + 0.5 * np.exp(-0.5 * ((t - 60) / 10) ** 2)
    V = noise_covariance(W, sig2, np.outer(d - d[3], dt_nu), share=rng.random(n_chan) + 0.5)
    assert np.allclose(V, V.T)
    assert np.isclose(V[3, 3], sigma_S(W[3], sig2) ** 2)


def test_constant_noise_diagonal_is_sigma_S_squared():
    """With constant sigma_t^2, the shifts do not change the variance: V_jj = sigma_S^2 at every
    trial DM."""
    W = smooth_rows(5)
    d = 0.05 * np.arange(5)
    sig2 = np.full(N, 2.0)
    V = noise_covariance(W, sig2, np.outer(d - d[0], dt_nu))
    sigma_S_squared = []
    for w in W:
        sigma_S_squared.append(sigma_S(w, sig2) ** 2)
    assert np.allclose(np.diag(V), sigma_S_squared)


def test_matches_monte_carlo_with_self_noise():
    """Empirical Cov[e_par(d_j), e_par(d_l)] from white channel noise whose variance rises during
    the burst (as self-noise does), set in the dynamic spectrum at d_ref and dedispersed as
    x(t; d) = sum_nu n_nu(t + (d - d_ref) dt_nu). Unevenly spaced trial DMs."""
    d = np.array([-0.15, 0.0, 0.05, 0.2])
    d_ref = 0.0
    W = smooth_rows(d.size)
    share = np.linspace(0.5, 1.5, n_chan)
    share = share / share.sum()
    sig2 = 1 + 4 * np.exp(-0.5 * ((t - 60) / 4) ** 2)

    # Each channel's shift at each trial DM as a phase ramp: shape (trial DM, channel, frequency).
    shifts = (d - d_ref)[:, None] * dt_nu[None, :]
    ramp = np.exp(2j * np.pi * FR[None, None, :] * shifts[:, :, None])

    e = []
    for _ in range(4000):
        n = rng.normal(size=(n_chan, N)) * np.sqrt(share[:, None] * sig2[None, :])

        # The noise profile at each trial DM, and e_par(d_j) = sum_t w_t n_t.
        x = np.fft.irfft((np.fft.rfft(n, axis=1)[None] * ramp).sum(axis=1), N)
        e.append((W * x).sum(axis=1))
    V_mc = np.cov(np.array(e).T)

    V = noise_covariance(W, sig2, np.outer(d - d_ref, dt_nu), share)
    assert np.allclose(V, V_mc, rtol=0.1, atol=0.05 * V.max())

    # The quantity sigma_DM needs: std[c1] of a parabola fitted over these trial DMs (Eq. varc1),
    # within 3% (the Monte Carlo error is about 1%). Averaging sigma_t^2 over the two trial DMs'
    # frames instead, as an earlier version did, is about 10% low here.
    A = np.vstack([np.ones(d.size), d, d**2]).T
    b1 = np.linalg.pinv(A)[1]
    assert abs(np.sqrt(b1 @ V @ b1) / np.sqrt(b1 @ V_mc @ b1) - 1) < 0.03


def test_full_matrix_equals_diagonal():
    """Cov(n) given as the full matrix diag(sigma_t^2) gives the same V as its diagonal."""
    W = smooth_rows(5)
    delay = np.outer(0.05 * np.arange(5) - 0.1, dt_nu)
    sig2 = 1 + 0.5 * np.exp(-0.5 * ((t - 60) / 10) ** 2)
    assert np.allclose(noise_covariance(W, np.diag(sig2), delay),
                       noise_covariance(W, sig2, delay))


def test_matches_monte_carlo_correlated_noise():
    """Noise correlated between samples (exponential correlation over 3 samples), the same in
    every channel up to its share: Cov(m_nu) = s_nu Cov(n), with Cov(n) passed as a full matrix."""
    d = np.array([-0.1, 0.0, 0.1, 0.2])
    d_ref = 0.0
    W = smooth_rows(d.size)
    cov_n = 2.0 * np.exp(-np.abs(t[:, None] - t[None, :]) / 3.0)
    chol = np.linalg.cholesky(cov_n)
    share = np.full(n_chan, 1.0 / n_chan)

    shifts = (d - d_ref)[:, None] * dt_nu[None, :]
    ramp = np.exp(2j * np.pi * FR[None, None, :] * shifts[:, :, None])

    e = []
    for _ in range(4000):
        # Each channel's noise has covariance s_nu Cov(n): unit white noise through the Cholesky
        # factor, scaled by sqrt(s_nu).
        n = (rng.normal(size=(n_chan, N)) @ chol.T) * np.sqrt(share[:, None])
        x = np.fft.irfft((np.fft.rfft(n, axis=1)[None] * ramp).sum(axis=1), N)
        e.append((W * x).sum(axis=1))
    V_mc = np.cov(np.array(e).T)

    V = noise_covariance(W, cov_n, np.outer(d - d_ref, dt_nu), share)
    A = np.vstack([np.ones(d.size), d, d**2]).T
    b1 = np.linalg.pinv(A)[1]
    assert np.allclose(V, V_mc, rtol=0.1, atol=0.05 * V.max())
    assert abs(np.sqrt(b1 @ V @ b1) / np.sqrt(b1 @ V_mc @ b1) - 1) < 0.03
