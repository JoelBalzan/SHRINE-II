"""Structure-maximising DM against scattering time (Section 5.1).

For the one- and two-component bursts of Section 2.1 regenerated with FIRES at tau = 0, 0.05, 0.1, 0.2
and 0.4 ms at 1 GHz (bursts/tau<TTT>_<tag>.npz with TTT = 100 tau/ms, from bursts/configs/fires_tau*.toml; otherwise
identical, with the same microshot realisation), and for each S/N of Section 4: k_c from S23's
procedure on the pilot realisations (sims.pilot, with the seeds of sims.CASES), and d_0 and the
predicted sigma_DM of the noiseless burst at that k_c. The offset d_0 - DM_true is given in pc cm^-3
and in units of sigma_DM. tau = 0 is the unscattered control.

The regenerated bursts use the present FIRES channel grid (336 channels centred on whole MHz, against
337 centred on half MHz for bursts/single.npz and double.npz), so the script also runs the paper's
bursts (burst "single" and "double", tau_ms None in the output) to compare with the tau = 0.2 ms rows;
d_0 agrees to within 0.002 pc cm^-3.

Writes results/tau_scan.json.

usage: python tau_scan.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402

TAUS = [0, 5, 10, 20, 40]  # in units of 0.01 ms


def main():
    rows = []
    for tag in ["single", "double"]:
        for tau in TAUS + ["paper"]:
            burst = tag if tau == "paper" else f"tau{tau:03d}_{tag}"
            I0, freq, time_ms = sims.load_burst(burst)
            dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
            for snr in sims.SNRS:
                _, seed = sims.CASES[(tag, snr)]
                sc = sims.channel_noise_sigma(I0, snr)
                p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed))
                rows.append(dict(tag=tag, tau_ms=None if tau == "paper" else tau / 100, burst=burst,
                                 snr=snr, kc=int(p["kc"]), d0=float(p["d0"]),
                                 sig_DM=float(p["sig_DM"]),
                                 d0_over_sig=float(p["d0"] / p["sig_DM"])))
                r = rows[-1]
                print(f"{burst:16s} S/N {snr:3d}: k_c {r['kc']:3d}, d0 {r['d0']:+.4f} pc cm^-3, "
                      f"sigma_DM {r['sig_DM']:.4f}, d0/sigma_DM {r['d0_over_sig']:+.2f}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "tau_scan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
