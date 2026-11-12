"""Does profiles.channel_delays() match what the dedispersion actually does to each coarse channel?

A delta function in the voltages (flat spectrum, so every channel sees it) is dedispersed by a trial
DM d, and the pulse's arrival time in each coarse channel is compared with the model
I(t; d) = sum_nu I_nu(t + d tau_nu), i.e. a shift of -d tau_nu, with tau_nu measured from f0.

The fit is moved = slope x model + offset, and the slope should be 1. An offset that is the same at
every trial DM does not matter for the covariance. One that grows with d does: it would mean tau_nu
is measured from a different frequency than the one the dedispersion holds fixed, which adds
(d_j - d_l) tau_ref to the lag between trial DMs in every channel. (Measured from the top of the
band, the offset at d = 0.05 would be about -28 us.)

The offset printed, about +1.8 us, is the harmless kind. At d = 0 the pulse fills one 1-us sample,
so the 5-sample running sum is flat over five samples and argmax picks the first, two samples early.
The offset stays at +1.8 us for d from 0.05 to 0.4.

usage: python check_delays.py
"""
import os
import sys

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shrine2.profiles import (NCHAN, channel_delays, channelise, dedispersion_phase,
                              fft_bin_frequencies)

# CRAFT's band, 2 ms of voltages, and the trial DM.
F0 = 1271.5
BW = 336
N_SAMPLES = 2 * 1000 * 336
DM = 0.05


def channel_power(F, freqs, d):
    """The power in each coarse channel after dedispersing the voltages by d, as (time in us,
    channel).

    Args:
        F: FFT of the voltages, shape (N_SAMPLES,).
        freqs: frequency of each FFT bin, MHz.
        d: trial delta-DM, pc cm^-3.
    Returns:
        |coarse channels|^2, shape (time in us, channel).
    """
    dedispersed = np.fft.ifft(F * dedispersion_phase(d, freqs))
    return np.abs(channelise(dedispersed)) ** 2


def arrival_time(S, c):
    """Arrival time of the pulse in one channel (peak of a 5-sample running sum).

    Args:
        S: channel power, shape (time, channel).
        c: channel index.
    Returns:
        time of the pulse in channel c, in 1-us samples.
    """
    running_sum = np.convolve(S[:, c], np.ones(5), mode="same")
    return int(np.argmax(running_sum))


def main():
    """Measure each channel's shift between d = 0 and d = DM and fit it against the model."""
    # A delta function at t = 1000 us.
    x = np.zeros(N_SAMPLES, complex)
    x[1000 * 336 + 100] = 1.0
    freqs = fft_bin_frequencies(F0, BW, N_SAMPLES)
    F = np.fft.fft(x)

    power_0 = channel_power(F, freqs, 0.0)
    power_d = channel_power(F, freqs, DM)

    # tau has entries 0, 1 for the two halves of channel 0, then channels 1 .. 335; compare the
    # channels 1 .. 334, so channel c is entry c + 1.
    tau, _ = channel_delays(F0, BW, NCHAN)
    chans = np.arange(1, NCHAN - 1)

    # How far the pulse moved in each channel between d = 0 and d = DM, in 1-us samples.
    shifts = []
    for c in chans:
        shift = arrival_time(power_d, c) - arrival_time(power_0, c)
        shifts.append(shift)
    moved = np.array(shifts)

    # The shift the model predicts in each channel, -DM tau_nu.
    model = -DM * tau[chans + 1]

    # moved = slope x model + offset: the slope should be 1, and the offset should not grow with DM
    # (see above).
    slope, offset = np.polyfit(model, moved, 1)
    rms = np.std(moved - slope * model - offset)

    print(f"channels 1, 168, 334: moved {moved[[0, 167, 332]]} us, "
          f"model {np.round(model[[0, 167, 332]], 1)} us")
    print(f"moved = {slope:.4f} x model + {offset:.2f} us; rms about the line {rms:.2f} us "
          "(1-us samples)")


if __name__ == "__main__":
    main()
