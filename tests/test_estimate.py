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

    def negative_covariance(W, sig2, d, d_ref, dt_nu, share=None):
        return -np.eye(W.shape[0])

    monkeypatch.setattr(shrine2.estimate, "noise_covariance", negative_covariance)
    d = np.linspace(-0.12, 0.12, 121)
    res = dm_uncertainty(dedisperse(I0, d), d, 30, dt_nu)
    assert not res["ok"]
    assert "not positive" in res["why"]
