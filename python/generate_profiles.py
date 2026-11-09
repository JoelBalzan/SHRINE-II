#!/usr/bin/env python3
"""I(DM, t) for an FRB from CRAFT X/Y voltages. Same flags, inputs and outputs as SHRINE's generate_profiles.py.

Inputs  (in the working directory)
    {label}_X_t_{DM}.npy, {label}_Y_t_{DM}.npy   voltages, already dedispersed to DM
    -I <file>                                    an I(channel, t) array, used only to find the burst for the crop
Outputs
    {label}_DMs.npy / .dat                       the trial delta-DMs
    {label}_I_{dt}us.npy / .dat                  I(DM, t), one row per trial delta-DM
    {label}_profile_summaryfile.txt
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run from a checkout without installing
from shrine2.profiles import crop, make_profiles, one_sample_dm_step, zapped_channels


def trial_dms(a, zap=()):
    """The trial delta-DMs from the flags.

    As SHRINE: --cDM evenly spaced values if --cDM > 0, otherwise -L, -L + --dDM, ... < -H.
    With --dDM_auto, the step is instead the one-sample DM step (profiles.one_sample_dm_step): the DM change that
    moves the bottom of the band by one output sample relative to the top.

    Args:
        a: the parsed flags; uses -L, -H, --dDM, --cDM, --dDM_auto, and -t, -f, -b for --dDM_auto.
        zap: the zapped channels (they can change the band's extent, and so the one-sample DM step).
    Returns:
        the trial delta-DMs, shape (n_d,), pc cm^-3.
    """
    if a.dDM_auto:
        step = one_sample_dm_step(a.f0, a.bw, a.dt, zap=zap)
        d = np.arange(a.DM_low, a.DM_high, step)
        print(f"DM step set to {step:.4g} pc/cm3 (one {a.dt}-us sample across the band): {d.size} trial DMs")
        if d.size > 2000:
            print("That is a lot of trial DMs: narrow -L/-H about the burst's DM to save time.")
        return d
    return np.linspace(a.DM_low, a.DM_high, a.cDM) if a.cDM > 0 else np.arange(a.DM_low, a.DM_high, a.dDM)


def main():
    """Read the voltages, crop them, and write I(DM, t) for the trial DMs."""
    a = get_args()
    X = np.load(f"{a.label}_X_t_{a.DM}.npy", mmap_mode="r")
    Y = np.load(f"{a.label}_Y_t_{a.DM}.npy", mmap_mode="r")
    I = np.load(a.I, mmap_mode="r")
    X, Y = crop(X, Y, I, a.crop_dur, a.force_peak)

    zap = zapped_channels(a.zap, a.zap_edges)
    d = trial_dms(a, zap)
    I_dt = make_profiles(X, Y, d, a.f0, a.bw, a.dt, zap)

    np.save(f"{a.label}_DMs.npy", d)
    np.save(f"{a.label}_I_{a.dt}us.npy", I_dt)
    np.savetxt(f"{a.label}_DMs.dat", d)
    np.savetxt(f"{a.label}_I_{a.dt}us.dat", I_dt)
    with open(f"{a.label}_zap.txt", "w") as out:        # read by maximise_structure.py for the channel delays
        out.write(" ".join(str(c) for c in zap) + "\n")
    with open(f"{a.label}_profile_summaryfile.txt", "w") as out:
        out.write("//begin generate_profiles summary//\n/*\n")
        out.write(f"X and Y files were of length {len(X)}\n")
        out.write(f"DM range was from {d[0]} to {d[-1]}\n")
        out.write(f"DM range was made up of {len(d)} steps of size {d[1] - d[0]}\n")
        if a.dDM_auto:
            out.write(f"DM step was set automatically (--dDM_auto): one {a.dt} us sample across the band\n")
        out.write(f"Zapped channels: {' '.join(str(c) for c in zap) or 'none'}\n")
        out.write(f"I profile was of shape {I_dt.shape}\n")
        out.write("*/\n//end generate_profiles summary//\n\n\n")


def get_args():
    """The command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    p = ArgumentParser(description="Generates an I(DM,t) profile for a given FRB.",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-l", "--label", type=str, help="FRB label")
    p.add_argument("-d", "--DM", type=str, help="Dispersion measure to which the input has been dedispersed in pc/cm3")
    p.add_argument("-r", "--DM_range", type=float, default=1, help="Unused; kept for compatibility with SHRINE")
    p.add_argument("-L", "--DM_low", type=float, default=-5, help="Lower bound of delta DM range")
    p.add_argument("-H", "--DM_high", type=float, default=5, help="Upper bound of delta DM range")
    p.add_argument("--dDM", type=float, default=0.1, help="Delta DM step size in pc/cm3")
    p.add_argument("--cDM", type=int, default=0, help="Number of DM steps to divide the range into")
    p.add_argument("--dDM_auto", action="store_true", default=False,
                   help="Set the delta DM step to the one-sample DM step for -t, -f and -b (ignores --dDM and --cDM)")
    p.add_argument("-t", "--dt", type=int, default=1, help="Time resolution to return in us")
    p.add_argument("-f", "--f0", type=float, help="Central frequency of observation in MHz")
    p.add_argument("-b", "--bw", type=int, default=336, help="Bandwidth of observation in MHz")
    p.add_argument("--crop_dur", type=int, default=10, help="Duration of the crop in ms")
    p.add_argument("-I", type=str, help="I profile filename (for doing the crop)")
    p.add_argument("--force_peak", type=int, default=None, help="Force the peak to be at this time step (in ms)")
    p.add_argument("--zap", nargs="+", default=[],
                   help="Coarse channels to leave out: indices and inclusive ranges, e.g. --zap 0 100-110 335 "
                        "(SHRINE always left out 335)")
    p.add_argument("--zap_edges", type=int, default=0,
                   help="Leave out the first and last N channel indices (1 = channels 0 and 335)")
    return p.parse_args()


if __name__ == "__main__":
    main()
