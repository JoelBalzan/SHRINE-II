"""End to end on a toy burst: the quoted sigma_DM matches the scatter of d_fit over noise
realisations (Section 4)."""
import numpy as np

from shrine2 import choose_kc, dm_uncertainty

rng = np.random.default_rng(3)

# Profiles of N samples summed from n_chan channels, with these delays (samples per unit DM).
N = 256
n_chan = 32
t = np.arange(N)
dt_nu = np.linspace(-40, 40, n_chan)
FR = np.fft.rfftfreq(N)

# The same two-component burst in every channel, at the true DM (d = 0).
burst = np.exp(-0.5 * ((t - 128) / 3) ** 2) + 0.6 * np.exp(-0.5 * ((t - 140) / 5) ** 2)
I0 = np.tile(burst, (n_chan, 1))


def dedisperse(dyn, d):
    """x(t; d_j) = sum_nu I_nu(t + d_j dt_nu), shifts applied as Fourier phase ramps.

    Args:
        dyn: dynamic spectrum, shape (n_chan, N).
        d: trial DMs, shape (n_d,).
    Returns:
        the profiles, shape (n_d, N).
    """
    ramp = np.exp(2j * np.pi * FR[None, None, :] * (d[:, None] * dt_nu[None, :])[:, :, None])
    return np.fft.irfft((np.fft.rfft(dyn, axis=1)[None] * ramp).sum(axis=1), N)


def test_sigma_DM_matches_scatter():
    d = np.linspace(-0.12, 0.12, 121)
    P0 = dedisperse(I0, d)
    sig_chan = 0.12

    # k_c from S23's procedure on one realisation, then held fixed.
    kc = choose_kc(P0 + dedisperse(rng.normal(0, sig_chan, I0.shape), d))

    d_fit = []
    sig_DM = []
    for _ in range(150):
        res = dm_uncertainty(P0 + dedisperse(rng.normal(0, sig_chan, I0.shape), d), d, kc, dt_nu)
        if res["ok"]:
            d_fit.append(res["d_fit"])
            sig_DM.append(res["sig_DM"])
    assert len(d_fit) > 140

    ratio = np.median(sig_DM) / np.std(d_fit)
    assert 0.85 < ratio < 1.15, ratio

    # d_fit unbiased, to within its sampling error and a small allowance.
    assert abs(np.mean(d_fit)) < 3 * np.std(d_fit) / np.sqrt(len(d_fit)) + 0.2 * np.std(d_fit)


def test_non_positive_variance_is_a_failure(monkeypatch):
    """A V that is not a valid covariance gives no sigma_DM, rather than sigma_DM = 0."""
    import shrine2.estimate

    def negative_covariance(W, cov_n, delay, share=None):
        return -np.eye(W.shape[0])

    monkeypatch.setattr(shrine2.estimate, "noise_covariance", negative_covariance)
    d = np.linspace(-0.12, 0.12, 121)
    res = dm_uncertainty(dedisperse(I0, d), d, 30, dt_nu)
    assert not res["ok"]
    assert "not positive" in res["why"]


def test_cov_n_diagonal_matches_default():
    """cov_n = diag(sigma_t^2), given as a vector or a matrix, reproduces the default (white noise).

    Only e_ratio differs for the matrix: from a full Cov(n) it is exact, whereas from sigma_t^2 it
    takes the noise as white with the mean of sigma_t^2 (noise.filtered_noise_ratio)."""
    d = np.linspace(-0.12, 0.12, 121)
    P = dedisperse(I0, d) + dedisperse(rng.normal(0, 0.12, I0.shape), d)
    res = dm_uncertainty(P, d, 30, dt_nu)
    assert res["ok"]
    for cov_n in [res["sig2"], np.diag(res["sig2"])]:
        given = dm_uncertainty(P, d, 30, dt_nu, cov_n=cov_n)
        assert given["ok"]
        for key in ["d_fit", "sig_DM", "sig_S", "c2_err", "n_win"]:
            assert np.isclose(given[key], res[key], rtol=1e-10, atol=0), key
        rtol = 1e-10 if cov_n.ndim == 1 else 0.05
        assert np.isclose(given["e_ratio"], res["e_ratio"], rtol=rtol, atol=0)


def test_sigma_DM_matches_scatter_correlated_noise():
    """With noise correlated between samples and its full Cov(n) given, the quoted sigma_DM matches
    the scatter of d_fit (Section 5.3)."""
    d = np.linspace(-0.12, 0.12, 121)
    P0 = dedisperse(I0, d)
    sig_chan = 0.12

    # AR(1)-like noise, rho = 0.5, unit variance per sample, made circularly (so that the circular
    # dedispersion leaves its covariance unchanged): power spectrum H(f) on the rfft frequencies.
    rho = 0.5
    H = (1 - rho**2) / (1 + rho**2 - 2 * rho * np.cos(2 * np.pi * FR))
    H /= np.fft.irfft(H, N)[0]
    autocov = np.fft.irfft(H, N)
    cov_n = n_chan * sig_chan**2 * autocov[np.abs(t[:, None] - t[None, :])]

    def noise():
        white = rng.normal(0, sig_chan, I0.shape)
        return np.fft.irfft(np.fft.rfft(white, axis=1) * np.sqrt(H), N, axis=1)

    kc = choose_kc(P0 + dedisperse(noise(), d))
    d_fit, sig_DM = [], []
    for _ in range(150):
        res = dm_uncertainty(P0 + dedisperse(noise(), d), d, kc, dt_nu, cov_n=cov_n)
        if res["ok"]:
            d_fit.append(res["d_fit"])
            sig_DM.append(res["sig_DM"])
    assert len(d_fit) > 140
    d_fit, sig_DM = np.array(d_fit), np.array(sig_DM)

    # Each realisation's d_fit against its own sigma_DM, about the mean d_fit.
    pull = np.std((d_fit - np.mean(d_fit)) / sig_DM)
    assert 0.85 < pull < 1.15, pull
    ratio = np.median(sig_DM) / np.std(d_fit)
    assert 0.8 < ratio < 1.2, ratio


def test_filtered_noise_ratio_full_covariance():
    """sqrt(<||e||^2>)/Sp from a full Cov(n) is sqrt(trace(D1 S Cov(n) S D1^T))/Sp."""
    from shrine2.noise import filtered_noise_ratio
    from shrine2.structure import forward_difference_matrix, smooth

    kc = 30
    cov_n = 0.3 * 0.6 ** np.abs(t[:, None] - t[None, :]) + np.diag(np.linspace(0.5, 2, N))
    D1S = forward_difference_matrix(N) @ smooth(np.eye(N), kc)
    direct = np.sqrt(np.trace(D1S @ cov_n @ D1S.T)) / 7.0
    assert np.isclose(filtered_noise_ratio(cov_n, kc, 7.0), direct, rtol=1e-10)

    # Constant variance: the vector and matrix forms agree.
    sig2 = np.full(N, 1.7)
    assert np.isclose(filtered_noise_ratio(sig2, kc, 7.0), filtered_noise_ratio(np.diag(sig2), kc, 7.0),
                      rtol=1e-10)
