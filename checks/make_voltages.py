"""Synthetic CRAFT-like X/Y voltages (3 ns samples, 336 MHz) with a burst dispersed by a known
delta-DM.

usage: python make_voltages.py <seed> <dir> [<dir> ...]
Writes test_X_t_0.npy, test_Y_t_0.npy and test_I.npy into each directory.
"""
import os
import sys

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shrine2.profiles import dedispersion_phase, fft_bin_frequencies


def synthetic_voltages(seed, dDM_true=0.03, amp=0.5, f0=1271.5, bw=336, n_ms=3, t0_us=1500,
                       w_us=30):
    """Synthetic CRAFT-like complex X/Y voltages: white noise plus a noise-like burst with a
    Gaussian envelope, dispersed by dDM_true.

    Args:
        seed: random seed for the burst and the noise.
        dDM_true: delta-DM by which the burst is dispersed, pc cm^-3.
        amp: burst amplitude relative to the noise.
        f0, bw: centre frequency and bandwidth, MHz.
        n_ms: length of the series, ms.
        t0_us, w_us: time and width (Gaussian sigma) of the main component, us.
    Returns:
        X, Y: complex64 voltages, shape (n_ms * 1000 * 336,).
    """
    rng = np.random.default_rng(seed)
    n = n_ms * 1000 * 336
    t_us = np.arange(n) / 336

    # The burst's power envelope: the main component and a fainter one 60 us later.
    main_component = np.exp(-0.5 * ((t_us - t0_us) / w_us) ** 2)
    second_component = 0.5 * np.exp(-0.5 * ((t_us - t0_us - 60) / 15) ** 2)
    env = main_component + second_component

    def complex_noise():
        """Unit-variance complex Gaussian noise, shape (n,)."""
        return (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)

    # The phase that disperses the burst by dDM_true (the conjugate of the dedispersion phase).
    freqs = fft_bin_frequencies(f0, bw, n)
    disp = np.conj(dedispersion_phase(dDM_true, freqs))

    # Each polarisation: receiver noise plus its own noise-like burst, dispersed. The left-hand term
    # is evaluated first, which fixes the order of the random draws for a given seed.
    X = complex_noise() + np.fft.ifft(np.fft.fft(amp * np.sqrt(env) * complex_noise()) * disp)
    Y = complex_noise() + np.fft.ifft(np.fft.fft(amp * np.sqrt(env) * complex_noise()) * disp)
    return X.astype(np.complex64), Y.astype(np.complex64)


def main():
    """Write the voltages for the seed given into each directory given."""
    X, Y = synthetic_voltages(int(sys.argv[1]))
    for directory in sys.argv[2:]:
        np.save(f"{directory}/test_X_t_0.npy", X)
        np.save(f"{directory}/test_Y_t_0.npy", Y)

        # Only used for the crop. It is flat, so give --force_peak (without it the crop is centred
        # on 0 ms and runs off the start).
        np.save(f"{directory}/test_I.npy", np.ones((336, 3000)))


if __name__ == "__main__":
    main()
