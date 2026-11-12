"""shrine2.snr.maximise_sn against SHRINE's own maximise_sn.py, on {label}_DMs.npy and
{label}_I_{dt}us.npy in the working directory (e.g. from generate_profiles.py run on
checks/make_voltages.py output).

usage: SHRINE=<path to SHRINE/python> python compare_sn_shrine.py <label> <dt>
Runs SHRINE's script (writing its outputs into the working directory), then prints both results;
the maximum S/N is also recomputed with SHRINE's boxcar off-by-one put back, which should equal
SHRINE's to rounding.
"""
import os
import subprocess
import sys

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shrine2.snr import maximise_sn
from shrine2.structure import choose_kc


def main():
    """Run SHRINE's maximise_sn.py, then SHRINE-II's, and print both."""
    label = sys.argv[1]
    dt = int(sys.argv[2])
    shrine = os.environ["SHRINE"]

    # SHRINE's own script, which writes {label}_SN_summaryfile.txt.
    command = [sys.executable, os.path.join(shrine, "maximise_sn.py"), "-l", label, "-t", str(dt)]
    subprocess.run(command, check=True, env=dict(os.environ, PYTHONPATH=shrine))
    with open(f"{label}_SN_summaryfile.txt") as summary_file:
        print(summary_file.read())

    # SHRINE-II, with S23's k_c as SHRINE's maximise_sn uses.
    d = np.load(f"{label}_DMs.npy")
    I = np.load(f"{label}_I_{dt}us.npy")
    sn = maximise_sn(I, d, choose_kc(I, extend=False), dt)
    print(f"SHRINE-II: index {sn['j_max']}, bounds {sn['lo']} - {sn['hi']}, "
          f"window {sn['start']} to {sn['start'] + sn['length']}, max S/N {sn['SN_max']}")

    # SHRINE's boxcar: L + 1 samples summed and divided by sqrt(L).
    X = sn["Sx"] - sn["Sx"].min()
    C = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, axis=1)], axis=1)
    best = -np.inf
    for L in range(1, X.shape[1] - 1):
        sn_L = (C[:, L + 1:] - C[:, :-(L + 1)]) / (sn["sigma"] * np.sqrt(L))
        best = max(best, sn_L.max())
    print(f"SHRINE-II with SHRINE's off-by-one: max S/N {best}")


if __name__ == "__main__":
    main()
