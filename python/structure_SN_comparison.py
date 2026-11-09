#!/usr/bin/env python3
"""Smoothed profiles at the structure- and S/N-maximising delta DMs. Same flags as SHRINE's structure_SN_comparison.py.

Writes {label}_structure_SN_comparison.png. The smoothed profiles are {label}_I_smooth_{dt}us.npy from
maximise_structure.py (or {label}_I_smooth.npy from maximise_sn.py -s).
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run from a checkout without installing
from shrine2.plots import plot_structure_sn


def main():
    """Read the trial DMs and smoothed profiles and plot the two rows."""
    a = get_args()
    d, Sx = np.load(a.DM_data), np.load(a.smooth_data)
    plot_structure_sn(Sx, d, a.i_st, a.i_sn, a.dt, f"{a.label}_structure_SN_comparison.png")


def get_args():
    """The command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    p = ArgumentParser(description="Plots a comparison of the smoothed signal at the structure and S/N maximising delta DMs.",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-l", "--label", type=str, help="FRB label")
    p.add_argument("-t", "--dt", type=int, default=1, help="Time resolution of inputs in us.")
    p.add_argument("-i_st", type=int, help="Index of the structure maximising delta DM")
    p.add_argument("-i_sn", type=int, help="Index of the signal to noise ratio maximising delta DM")
    p.add_argument("-d", "--DM_data", type=str, help="Path to the DM data.")
    p.add_argument("-I", "--I_data", type=str, help="Path to the I data (unused; kept for compatibility with SHRINE).")
    p.add_argument("-s", "--smooth_data", type=str, help="Path to the smoothed I data.")
    return p.parse_args()


if __name__ == "__main__":
    main()
