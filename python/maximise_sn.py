#!/usr/bin/env python3
"""The S/N-maximising delta DM of an FRB, from I(DM, t). Same flags, inputs and outputs as SHRINE's
maximise_sn.py.

A standalone alternative to maximise_structure.py, and the one to use for very low S/N bursts,
where structure maximisation does not work well (see shrine2/snr.py for the method and what was
fixed relative to SHRINE). Run it on the output of generate_profiles.py, or run
optimise_dm.py --sn_only to go straight from the voltages.

Inputs  (from generate_profiles.py, in the working directory)
    {label}_DMs.npy, {label}_I_{dt}us.npy
Outputs
    {label}_SN_summaryfile.txt   the S/N-maximising delta DM, the range where S/N is within 1 of
                                 its maximum, the boxcar
    {label}_SN_DMs.dat           trial delta-DMs relative to the S/N-maximising one
    {label}_SN.png, {label}_Max_SN_window.png, {label}_{dt}us_I_SN_max.png
    with -s: {label}_I_smooth.npy and DM.txt (the row of I(DM, t) at the maximum), as SHRINE
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shrine2 import choose_kc, plots
from shrine2.snr import maximise_sn
from shrine2.summary import write_sn_summary


def kc_for_sn(I, args):
    """The filter cutoff k_c for smoothing the profiles before the boxcar search: -kc if given,
    otherwise the automatic choice (choose_kc).

    Args:
        I: I(d_j, t), shape (n_d, N).
        args: the parsed flags; uses -kc and --s23_kc.
    Returns:
        the filter cutoff k_c.
    """
    if args.force_kc is not None:
        return args.force_kc
    return choose_kc(I, extend=not args.s23_kc)


def write_sn_outputs(sn, I, label, dt, save_all):
    """Write every output file of SHRINE's maximise_sn.py for a result of snr.maximise_sn.

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

    plots.plot_sn_vs_dm(sn["d"], sn["SN"], f"{label}_SN.png")
    plots.plot_sn_window(sn["Sx"][j], sn["start"], sn["length"], dt, f"{label}_Max_SN_window.png")
    plots.plot_profile_and_smoothed(I[j], sn["kc"], dt, f"{label}_{dt}us_I_SN_max.png")

    if save_all:
        np.save(f"{label}_I_smooth.npy", sn["Sx"])
        with open("DM.txt", "w") as out:
            out.write(f"{j}")


def print_sn_result(sn):
    """Print the S/N-maximising delta DM, its range and the k_c used.

    Args:
        sn: the dict returned by snr.maximise_sn.
    """
    d = sn["d"]
    j = sn["j_max"]
    print(f"S/N-maximising delta DM = {sn['d_sn']:.6g} "
          f"(+{d[sn['hi']] - d[j]:.3g} -{d[j] - d[sn['lo']]:.3g}, S/N {sn['SN_max']:.1f}), "
          f"k_c = {sn['kc']}")
    if not (sn["lo_found"] and sn["hi_found"]):
        print("Warning: the S/N does not fall by 1 within the trial DMs on one side: "
              "widen -L/-H for that bound")


def main():
    """Read the profiles written by generate_profiles.py, find the S/N-maximising DM and write
    SHRINE's maximise_sn outputs."""
    args = get_args()
    d = np.load(f"{args.label}_DMs.npy")
    I = np.load(f"{args.label}_I_{args.dt}us.npy")

    kc = kc_for_sn(I, args)
    sn = maximise_sn(I, d, kc, args.dt)
    write_sn_outputs(sn, I, args.label, args.dt, args.save_all)
    print_sn_result(sn)


def get_args():
    """Parse the command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    parser = ArgumentParser(
        description="Calculates signal to noise ratio maximising delta DM for pre-dedispersed "
                    "data.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-l", "--label", type=str, help="FRB label")
    parser.add_argument("-t", "--dt", type=int, default=1, help="Time resolution of inputs in us.")
    parser.add_argument("-s", "--save_all", action="store_true", default=False,
                        help="Save all plots.")
    parser.add_argument("-kc", "--force_kc", type=int, default=None, help="Value of kc to use.")
    parser.add_argument("--s23_kc", action="store_true", default=False,
                        help="Choose kc with S23's procedure only, without the extended search of "
                             "the paper's Appendix 1.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
