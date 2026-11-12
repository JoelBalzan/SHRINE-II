"""End to end on CRAFT-like voltages: profiles as SHRINE, then Section 3.3 with the SHRINE channel
model. Is the median quoted sigma_DM equal to the scatter of d_fit over noise realisations?

usage: python mc_voltages.py <burst amplitude> <realisations> [<zap_edges>]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_voltages import synthetic_voltages
from shrine2 import choose_kc, dm_uncertainty
from shrine2.profiles import channel_delays, crop_voltages, make_profiles, parse_zapped_channels

# CRAFT's band, the output time resolution (us) and the delta-DM the burst is dispersed by.
F0 = 1271.5
BW = 336
DT = 5
D_TRUE = 0.03

# The zapped channels (optional third argument: --zap_edges), and the channel model to match.
if len(sys.argv) > 3:
    ZAP = parse_zapped_channels(zap_edges=int(sys.argv[3]))
else:
    ZAP = parse_zapped_channels(zap_edges=0)
TAU, SHARE = channel_delays(F0, BW, zap=ZAP)

# An I(channel, t) array for crop_voltages(); not used, since the peak is given.
I_DUMMY = np.ones((336, 2000))


def realisation_profiles(seed, d, amp):
    """The profiles I(d_j, t) of one noise realisation: make synthetic voltages, crop them and form
    the profiles as generate_profiles.py does.

    Args:
        seed: random seed of the realisation.
        d: trial delta-DMs, shape (n_d,).
        amp: burst amplitude relative to the noise.
    Returns:
        I(d_j, t), shape (n_d, N).
    """
    X, Y = synthetic_voltages(seed, dDM_true=D_TRUE, amp=amp, n_ms=2, t0_us=1000)
    X, Y = crop_voltages(X, Y, I_DUMMY, 1, peak_ms=1)
    return make_profiles(X, Y, d, F0, BW, DT, ZAP)


def fit_one_realisation(job):
    """Run structure maximisation (dm_uncertainty) on one noise realisation (in parallel, via the
    Pool).

    Args:
        job: (seed, d, kc, amp) for one realisation.
    Returns:
        (d_fit, sigma_DM, n_win, c2_err, e_ratio), or None if the fit failed.
    """
    seed, d, kc, amp = job
    res = dm_uncertainty(realisation_profiles(seed, d, amp), d, kc, TAU / DT, SHARE)
    if not res["ok"]:
        return None
    return (res["d_fit"], res["sig_DM"], res["n_win"], res["c2_err"], res["e_ratio"])


def main():
    """Choose k_c and the trial DMs from a pilot realisation, then run the Monte Carlo and print the
    median sigma_DM over the scatter of d_fit."""
    amp = float(sys.argv[1])
    n_real = int(sys.argv[2])

    # Pilot on a coarse grid: k_c, and sigma_DM for the step of the Monte Carlo's grid.
    d_pilot = np.arange(-0.2, 0.26, 0.01)
    P = realisation_profiles(1000, d_pilot, amp)
    kc = choose_kc(P)
    pilot = dm_uncertainty(P, d_pilot, kc, TAU / DT, SHARE)
    pilot_values = {}
    for key in ["d_fit", "sig_DM", "sig_S", "e_ratio"]:
        if key in pilot:
            if np.isscalar(pilot[key]):
                pilot_values[key] = round(float(pilot[key]), 5)
            else:
                pilot_values[key] = None
    print("pilot:", pilot_values, "kc", kc, pilot["why"])

    # The Monte Carlo's trial DMs: step sigma_DM/5, 50 steps either side of the true delta-DM.
    step = pilot["sig_DM"] / 5
    d = D_TRUE + step * np.arange(-50, 51)

    # One job per noise realisation, run two at a time.
    jobs = []
    for seed in range(n_real):
        jobs.append((seed, d, kc, amp))

    t_start = time.time()
    with Pool(2) as pool:
        results = pool.map(fit_one_realisation, jobs)

    # Keep the realisations whose fit succeeded.
    successful = []
    for r in results:
        if r is not None:
            successful.append(r)
    ok = np.array(successful)
    d_fit = ok[:, 0]
    sig_DM = ok[:, 1]

    ratio = np.median(sig_DM) / d_fit.std()
    ratio_error = 1 / np.sqrt(2 * len(ok))
    bias = (d_fit.mean() - D_TRUE) / d_fit.std()
    print(f"amp {amp}: {len(ok)}/{n_real} ok, {time.time() - t_start:.0f}s; "
          f"median sigma_DM / scatter = {ratio:.3f} +- {ratio_error:.3f}; "
          f"mean d_fit - d_true = {bias:+.2f} scatter; "
          f"n_win median {np.median(ok[:, 2]):.0f}; c2err {np.median(ok[:, 3]):.2f}; "
          f"e_ratio {np.median(ok[:, 4]):.2f}")


if __name__ == "__main__":
    main()
