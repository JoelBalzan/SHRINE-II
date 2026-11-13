"""Drifting sub-bursts (Section 5.1): the structure-maximising DM of the noiseless bursts against
the sub-burst separation t_sep.

For each drift burst (three sub-bursts, t_sep = 0 ... 0.8 ms; bursts/drift*.npz) and k_c set at
S/N = 15, 40 and 200 (sims.pilot, seed 1): d_0 and the predicted sigma_DM of the noiseless burst
(sims.noiseless_fit, with the noise level of that S/N). Also the DM that maximises the peak of the
profile, and, for t_sep = 0.35 ms, Sp(d) with the k_c of S/N 15 and of S/N 200 (Figure 9c).

Writes results/drift_results.json and results/drift_curves.npz.

usage: python run_drift.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from shrine2.structure import structure_parameter  # noqa: E402

T_SEPS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]  # ms
SNRS = [15, 40, 200]
SEED = 1


def drift_tag(t_sep):
    """The burst's tag, e.g. drift035 for t_sep = 0.35 ms."""
    return f"drift{int(round(t_sep * 100)):03d}"


def main():
    rows = []
    kc_for_curves = {}
    for t_sep in T_SEPS:
        tag = drift_tag(t_sep)
        I0, freq, time = sims.load_burst(tag)
        dt_nu = sims.channel_delays(freq, time[1] - time[0])

        # The DM that maximises the peak of the frequency-summed profile.
        d_peak = np.linspace(-0.1, 0.6, 1401)
        P = sims.dedisperse_grid(I0, d_peak[0], d_peak[1] - d_peak[0], d_peak.size, dt_nu)
        dm_peak = float(d_peak[np.argmax(P.max(axis=1))])

        row = dict(t_sep=t_sep, tag=tag, dm_peak=dm_peak, snr={})
        for snr in SNRS:
            sc = sims.channel_noise_sigma(I0, snr)
            p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(SEED))
            row["snr"][str(snr)] = dict(kc=p["kc"], d0=p["d0"], sig_DM=p["sig_DM"], sc=sc)
            print(f"t_sep {t_sep:.2f} ms, S/N {snr:3d}: k_c {p['kc']:3d}, d0 {p['d0']:+.4f}, "
                  f"sigma_DM {p['sig_DM']:.5f}, d0/sigma_DM {p['d0'] / p['sig_DM']:+.1f}; "
                  f"peak-maximising DM {dm_peak:+.3f}", flush=True)
            if t_sep == 0.35:
                kc_for_curves[snr] = p["kc"]
        rows.append(row)

    # Sp(d) of the t_sep = 0.35 ms burst with the k_c of S/N 15 and of S/N 200.
    I0, freq, time = sims.load_burst(drift_tag(0.35))
    dt_nu = sims.channel_delays(freq, time[1] - time[0])
    d = np.linspace(-0.1, 0.45, 551)
    P = sims.dedisperse_grid(I0, d[0], d[1] - d[0], d.size, dt_nu)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "drift_results.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(HERE, "results", "drift_curves.npz"), d=d,
                        Sp_low=structure_parameter(P, kc_for_curves[15]),
                        Sp_high=structure_parameter(P, kc_for_curves[200]),
                        kc_low=kc_for_curves[15], kc_high=kc_for_curves[200])


if __name__ == "__main__":
    main()
