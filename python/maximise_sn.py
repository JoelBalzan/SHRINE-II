#!/usr/bin/env python3
"""The S/N-maximising delta DM of an FRB, from I(DM, t). Same flags, inputs and outputs as SHRINE's maximise_sn.py.

Not part of the paper's method; kept for comparison (see shrine2/snr.py for the method and what was fixed).

Inputs  (from generate_profiles.py, in the working directory)
    {label}_DMs.npy, {label}_I_{dt}us.npy
Outputs
    {label}_SN_summaryfile.txt   the S/N-maximising delta DM, the range where S/N is within 1 of its maximum, the boxcar
    {label}_SN_DMs.dat           trial delta-DMs relative to the S/N-maximising one
    {label}_SN.png, {label}_Max_SN_window.png, {label}_{dt}us_I_SN_max.png
    with -s: {label}_I_smooth.npy and DM.txt (the row of I(DM, t) at the maximum), as SHRINE
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run from a checkout without installing
from shrine2 import choose_kc, plots
from shrine2.snr import maximise_sn
from shrine2.summary import write_sn_summary


def write_outputs(sn, I, label, dt, save_all):
    """Everything SHRINE's maximise_sn.py writes, for a result of snr.maximise_sn.

    Args:
        sn: the dict returned by snr.maximise_sn.
        I: I(d_j, t), shape (n_d, N).
        label: FRB label (-l).
        dt: time resolution, us (-t).
        save_all: whether to write the -s outputs.
    """
    j = sn["j_max"]
    write_sn_summary(f"{label}_SN_summaryfile.txt", sn, dt)
    np.savetxt(f"{label}_SN_DMs.dat", sn["d"] - sn["d"][j])
    plots.plot_sn(sn["d"], sn["SN"], f"{label}_SN.png")
    plots.plot_sn_window(sn["Sx"][j], sn["start"], sn["length"], dt, f"{label}_Max_SN_window.png")
    plots.plot_profile(I[j], sn["kc"], dt, f"{label}_{dt}us_I_SN_max.png")
    if save_all:
        np.save(f"{label}_I_smooth.npy", sn["Sx"])
        with open("DM.txt", "w") as out:
            out.write(f"{j}")


def main():
    """Read I(DM, t), find the S/N-maximising DM and write SHRINE's maximise_sn outputs."""
    a = get_args()
    d = np.load(f"{a.label}_DMs.npy")
    I = np.load(f"{a.label}_I_{a.dt}us.npy")
    kc = a.force_kc if a.force_kc is not None else choose_kc(I, extend=not a.s23_kc)
    sn = maximise_sn(I, d, kc, a.dt)
    write_outputs(sn, I, a.label, a.dt, a.save_all)
    print(f"S/N-maximising delta DM = {sn['d_sn']:.6g} (S/N {sn['SN_max']:.1f}), k_c = {kc}")


def get_args():
    """The command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    p = ArgumentParser(description="Calculates signal to noise ratio maximising delta DM for pre-dedispersed data.",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-l", "--label", type=str, help="FRB label")
    p.add_argument("-t", "--dt", type=int, default=1, help="Time resolution of inputs in us.")
    p.add_argument("-s", "--save_all", action="store_true", default=False, help="Save all plots.")
    p.add_argument("-kc", "--force_kc", type=int, default=None, help="Value of kc to use.")
    p.add_argument("--s23_kc", action="store_true", default=False,
                   help="Choose kc with S23's procedure only, without the extended search of the paper's Appendix 1.")
    return p.parse_args()


if __name__ == "__main__":
    main()
