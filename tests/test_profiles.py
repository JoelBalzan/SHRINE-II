"""The channel model of profiles.py (SHRINE's generate_profiles)."""
import numpy as np
import pytest

from shrine2.profiles import (channel_delays, crop_voltages, one_sample_dm_step,
                              parse_zapped_channels)


def test_one_sample_dm_step():
    """CRAFT, 336 MHz at 1271.5 MHz: the band's delay spread is ~1.4 ms per unit DM, so
    ~7.1e-4 pc cm^-3 per us."""
    step = one_sample_dm_step(1271.5, 336, 1)
    assert np.isclose(step, 7.13e-4, rtol=0.01)
    assert np.isclose(one_sample_dm_step(1271.5, 336, 10), 10 * step)


def test_channel_shares():
    """Channel 0 enters as two halves of half a share each; zapped channels drop out."""
    tau, share = channel_delays(1271.5, 336, 336)
    assert tau.size == 337
    assert np.isclose(share.sum(), 336)

    # Low frequencies later than f0, high frequencies earlier.
    assert tau.max() > 0 > tau.min()

    tau, share = channel_delays(1271.5, 336, 336, zap=[0, 335])
    assert tau.size == 334
    assert np.isclose(share.sum(), 334)


def test_parse_zapped_channels():
    assert parse_zapped_channels(["0,100-102", "335"]).tolist() == [0, 100, 101, 102, 335]
    assert parse_zapped_channels(zap_edges=1).tolist() == [0, 335]
    assert parse_zapped_channels(zap_edges=3).tolist() == [0, 1, 2, 333, 334, 335]
    assert parse_zapped_channels(["100"], 1).tolist() == [0, 100, 335]
    assert parse_zapped_channels().size == 0


def test_crop_finds_a_burst_in_the_last_millisecond():
    """All of I is searched, including the last 1-ms bin when I holds a whole number of ms."""
    n_ms = 4
    X = np.arange(n_ms * 1000 * 336, dtype=complex)
    I = np.ones((8, n_ms * 1000))
    I[:, 3500] = 100.0                                   # the burst, in the last millisecond
    X_crop, _ = crop_voltages(X, X, I, 2)
    assert X_crop[0] == 2 * 1000 * 336                   # centred on 3 ms: samples from 2 ms


def test_crop_off_the_data_raises():
    X = np.zeros(3 * 1000 * 336, dtype=complex)
    I = np.ones((8, 3000))
    with pytest.raises(ValueError):
        crop_voltages(X, X, I, 2, peak_ms=0)                      # would start at -1 ms
    with pytest.raises(ValueError):
        crop_voltages(X, X, I, 4, peak_ms=2)                      # would end at 4 ms
