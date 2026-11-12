"""The S/N maximisation (shrine2/snr.py, as SHRINE's maximise_sn.py)."""
import numpy as np

from shrine2 import choose_kc
from shrine2.snr import best_boxcars, maximise_sn

rng = np.random.default_rng(4)


def test_best_boxcar_matches_brute_force():
    X = rng.normal(size=(3, 60))
    X[:, 20:27] += 2.0
    sn, start, length = best_boxcars(X, 1.3, 25)

    for r in range(3):
        # Every boxcar of 1 to 25 samples: (S/N, start, length) of the best.
        candidates = []
        for L in range(1, 26):
            for s in range(0, 60 - L + 1):
                candidates.append((X[r, s:s + L].sum() / (1.3 * np.sqrt(L)), s, L))
        best = max(candidates)

        assert np.isclose(sn[r], best[0])
        assert (start[r], length[r]) == best[1:]


def test_range_contains_the_true_DM():
    """A burst (S/N ~40) dedispersed over trial DMs: the true DM (d = 0) lies in the S/N range, and
    the S/N falls by 1 on both sides (it did in 60 of 60 noise realisations)."""
    N = 256
    n_chan = 32
    t = np.arange(N)
    dt_nu = np.linspace(-40, 40, n_chan)                 # samples per unit DM
    FR = np.fft.rfftfreq(N)

    # The same burst in every channel at the true DM, plus white channel noise.
    burst = np.exp(-0.5 * ((t - 128) / 3) ** 2) + 0.6 * np.exp(-0.5 * ((t - 140) / 5) ** 2)
    dyn = np.tile(burst, (n_chan, 1)) + rng.normal(0, 0.5, (n_chan, N))

    # Profiles at each trial DM, x(t; d_j) = sum_nu I_nu(t + d_j dt_nu), as Fourier phase ramps.
    d = np.linspace(-0.3, 0.3, 61)
    ramp = np.exp(2j * np.pi * FR[None, None, :] * (d[:, None] * dt_nu[None, :])[:, :, None])
    I = np.fft.irfft((np.fft.rfft(dyn, axis=1)[None] * ramp).sum(axis=1), N)

    sn = maximise_sn(I, d, choose_kc(I), dt=10)
    assert sn["lo_found"] and sn["hi_found"]
    assert d[sn["lo"]] < 0 < d[sn["hi"]]
