"""d_fit against the true DM (Section 5.1).

For the Section 4 Monte Carlo runs (results/mc_*.npz, trial DMs measured from the true DM, so
d_true = 0): the offset d_0 - d_true in units of sigma_MC, the fraction of realisations in which
d_fit +- k sigma_DM contains the true DM (k = 1, 2, 3), and the rms of d_fit - d_true over sigma_MC.
Writes results/dtrue_numbers.json.

usage: python dtrue_numbers.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402


def main():
    rows = []
    for (tag, snr) in sims.CASES:
        f = np.load(sims.results_path(f"mc_{tag}_snr{snr}.npz"))
        meta = json.loads(str(f["meta"]))
        d, s = f["d_fit"], f["sig_DM"]
        ok = np.isfinite(d) & np.isfinite(s)
        d, s = d[ok], s[ok]
        sig_mc = np.std(d - meta["d0"])
        z = np.abs(d) / s
        rows.append(dict(tag=tag, snr=snr, d0=meta["d0"], d0_over_sig_mc=float(meta["d0"] / sig_mc),
                         cover1=float(np.mean(z < 1)), cover2=float(np.mean(z < 2)),
                         cover3=float(np.mean(z < 3)),
                         rms_over_sig_mc=float(np.sqrt(np.mean(d**2)) / sig_mc)))
        print(f"{tag} S/N {snr}: d0/sigma_MC {rows[-1]['d0_over_sig_mc']:.1f}, true DM within "
              f"1/2/3 sigma_DM in {rows[-1]['cover1']:.3f}/{rows[-1]['cover2']:.3f}/"
              f"{rows[-1]['cover3']:.3f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "dtrue_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
