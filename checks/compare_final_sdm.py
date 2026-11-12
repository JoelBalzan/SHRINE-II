"""Check SHRINE-II's dm_uncertainty against final_sdm.FinalEstimator.quoted (the code behind the
paper's Section 4) on the same FIRES bursts, trial DMs and noise realisations.

d_fit, sigma_S, the fit window and c2 agree to rounding. sigma_DM (and c2err) differ by a few per cent
per realisation: final_sdm.py uses the earlier form of V, which averaged sigma_t^2 over the two trial
DMs' frames, whereas shrine2.covariance attaches it to the dynamic spectrum at d_max (paper App. 2).

usage: SHRINE_SIMS=<shrine_sims/validation_tau02> python compare_final_sdm.py
"""
import json
import os
import sys

import numpy as np

# The paper's simulations, whose code (val_core, final_sdm) reads VALDIR when imported.
VAL = os.environ.get("SHRINE_SIMS", "/mnt/Storage/FIRES/shrine_sims/validation_tau02")
os.environ["VALDIR"] = VAL
sys.path.insert(0, VAL)

# Run from a checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from val_core import Filter, channel_delays, load_burst
from final_sdm import FinalEstimator

import shrine2

# (burst, S/N) cases compared, and the number of noise realisations of each.
CASES = [("single", 15), ("single", 40), ("single", 200), ("double", 40)]
N_REALISATIONS = 15


def dedisperse(x, grid, dt_nu, FR, N):
    """Dedisperse a dynamic spectrum at every trial DM in grid and sum over channels, as the
    paper's simulations do (Fourier phase ramps).

    Args:
        x: dynamic spectrum, shape (n_chan, N).
        grid: trial DMs, shape (n_d,).
        dt_nu: each channel's delay per unit DM, samples, shape (n_chan,).
        FR: rfft frequencies of an N-sample series.
        N: number of samples per profile.
    Returns:
        I(d_j, t), shape (n_d, N).
    """
    X = np.fft.rfft(x, axis=1)
    rows = []
    for d in grid:
        # Each channel's spectrum shifted by its lag at this trial DM, then summed over channels.
        shifted = X * np.exp(2j * np.pi * FR[None] * (d * dt_nu)[:, None])
        rows.append(np.fft.irfft(shifted.sum(0), N))
    return np.array(rows)


def compare_one_case(tag, snr):
    """Run both estimators on N_REALISATIONS noise realisations of one case and print the
    largest differences.

    Args:
        tag: "single" or "double" (the burst).
        snr: the S/N of the case.
    """
    meta = json.loads(str(np.load(f"{VAL}/res/{tag}_snr{snr}.npz")["meta"]))
    kc = meta["kc"]
    d0 = meta["d0"]
    sc = meta["sc"]
    sig = meta["sig_pilot"]

    I0, f, t = load_burst(tag)
    N = t.size
    dt_nu = channel_delays(f, t[1] - t[0], 919.5)
    FR = np.fft.rfftfreq(N)

    # The paper's trial DMs: step sigma_DM/6 out to 15 sigma_DM either side of d0.
    step = sig / 6
    n_grid = 2 * int(np.ceil(15 * sig / step)) + 1
    grid = d0 - step * (n_grid // 2) + step * np.arange(n_grid)

    P0 = dedisperse(I0, grid, dt_nu, FR, N)
    estimator = FinalEstimator(Filter(N, kc), kc, N, dt_nu, grid)

    rng = np.random.default_rng(7)
    rows = []
    failures = []
    for _ in range(N_REALISATIONS):
        P = P0 + dedisperse(rng.normal(0, sc, I0.shape), grid, dt_nu, FR, N)
        S = Filter(N, kc).sp(P)
        q = estimator.quoted(S, P)
        s = shrine2.dm_uncertainty(P, grid, kc, dt_nu)
        if q is None or not s["ok"]:
            failures.append((q is None, s["ok"]))
            continue
        rows.append([
            abs(s["d_fit"] - q["dhat"]) / q["sdm"],
            abs(s["sig_DM"] / q["sdm"] - 1),
            abs(s["sig_S"] / q["sigS"] - 1),
            abs(s["c2_err"] / q["c2err"] - 1),
            int(s["n_win"] != q["npts"]),
        ])

    if not rows:
        print(f"{tag:6s} S/N {snr:3d}: no realisation succeeded in both; failures {failures}")
        return

    D = np.array(rows)
    print(f"{tag:6s} S/N {snr:3d}: {len(D)} realisations; "
          f"max |d_fit diff|/sigma_DM {D[:, 0].max():.1e}, "
          f"max rel diff sigma_DM {D[:, 1].max():.1e}, sigma_S {D[:, 2].max():.1e}, "
          f"c2err {D[:, 3].max():.1e}, "
          f"window size mismatches {int(D[:, 4].sum())}; failures {failures}")


def main():
    """Compare the two estimators on every case."""
    for tag, snr in CASES:
        compare_one_case(tag, snr)


if __name__ == "__main__":
    main()
