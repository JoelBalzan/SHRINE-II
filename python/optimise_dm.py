#!/usr/bin/env python3
"""generate_profiles.py and maximise_structure.py in one go, on the trial DMs given.

Takes the flags of both scripts, makes the profiles on the trial DMs given (-L, -H, --dDM, --cDM
or --dDM_auto) and writes the outputs of both. If the step is coarser than sigma_DM/3, or the fit
window holds fewer than three trial DMs, the summary file says so and suggests -L, -H and --dDM
for a rerun; the trial DMs are not refined automatically.

With --do_sn, also finds the S/N-maximising DM (maximise_sn.py) on the same profiles.
With --sn_only, finds only the S/N-maximising DM (no structure maximisation): the standalone S/N
mode, for very low S/N bursts.
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

# Run from a checkout without installing, with the other scripts importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_profiles import trial_dms, write_zap_file
from maximise_sn import kc_for_sn
from maximise_sn import print_sn_result
from maximise_sn import write_sn_outputs
from maximise_structure import run_structure_maximisation
from shrine2.profiles import crop_voltages, make_profiles, parse_zapped_channels
from shrine2.snr import maximise_sn


def main():
    """Make the profiles from the voltages, then run structure maximisation, S/N maximisation
    (--sn_only) or both (--do_sn), and write their outputs."""
    args = get_args()

    # The voltages, cropped about the burst (as generate_profiles.py).
    X = np.load(f"{args.label}_X_t_{args.DM}.npy", mmap_mode="r")
    Y = np.load(f"{args.label}_Y_t_{args.DM}.npy", mmap_mode="r")
    I_for_crop = np.load(args.I, mmap_mode="r")
    X, Y = crop_voltages(X, Y, I_for_crop, args.crop_dur, args.force_peak)

    # The profiles, written as generate_profiles.py writes them.
    zap = parse_zapped_channels(args.zap, args.zap_edges)
    d = trial_dms(args, zap)
    I = make_profiles(X, Y, d, args.f0, args.bw, args.dt, zap)
    np.save(f"{args.label}_DMs.npy", d)
    np.save(f"{args.label}_I_{args.dt}us.npy", I)
    write_zap_file(args.label, zap)

    # The standalone S/N mode.
    if args.sn_only:
        sn = maximise_sn(I, d, kc_for_sn(I, args), args.dt)
        write_sn_outputs(sn, I, args.label, args.dt, args.save_all)
        print_sn_result(sn)
        return

    res = run_structure_maximisation(I, d, args, zap)

    # The S/N-maximising DM too, smoothed with the same k_c.
    if args.do_sn:
        sn = maximise_sn(I, d, res["kc"], args.dt)
        write_sn_outputs(sn, I, args.label, args.dt, args.save_all)
        print_sn_result(sn)


def get_args():
    """Parse the command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    parser = ArgumentParser(
        description="Structure-maximising delta DM and its 1-sigma uncertainty from CRAFT "
                    "voltages.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-l", "--label", type=str, help="FRB label")
    parser.add_argument("-d", "--DM", type=str,
                        help="Dispersion measure to which the input has been dedispersed in pc/cm3")
    parser.add_argument("-L", "--DM_low", type=float, default=-5,
                        help="Lower bound of delta DM range")
    parser.add_argument("-H", "--DM_high", type=float, default=5,
                        help="Upper bound of delta DM range")
    parser.add_argument("--dDM", type=float, default=0.1, help="Delta DM step size in pc/cm3")
    parser.add_argument("--cDM", type=int, default=0,
                        help="Number of DM steps to divide the range into")
    parser.add_argument("--dDM_auto", action="store_true", default=False,
                        help="Set the delta DM step to the one-sample DM step for -t, -f and -b "
                             "(ignores --dDM and --cDM)")
    parser.add_argument("-t", "--dt", type=int, default=1, help="Time resolution in us")
    parser.add_argument("-f", "--f0", type=float, required=True,
                        help="Central frequency of observation in MHz")
    parser.add_argument("-b", "--bw", type=float, default=336,
                        help="Bandwidth of observation in MHz")
    parser.add_argument("--nchan", type=int, default=336,
                        help="Number of coarse channels formed from the voltages")
    parser.add_argument("--crop_dur", type=int, default=10, help="Duration of the crop in ms")
    parser.add_argument("-I", type=str, help="I profile filename (for doing the crop)")
    parser.add_argument("--force_peak", type=int, default=None,
                        help="Force the peak to be at this time step (in ms)")
    parser.add_argument("--zap", nargs="+", default=[],
                        help="Coarse channels to leave out: indices and inclusive ranges, e.g. "
                             "--zap 0 100-110 335 (SHRINE always left out 335)")
    parser.add_argument("--zap_edges", type=int, default=0,
                        help="Leave out the first and last N channel indices "
                             "(1 = channels 0 and 335)")
    parser.add_argument("-s", "--save_all", action="store_true", default=False,
                        help="Save all plots.")
    parser.add_argument("-kc", "--force_kc", type=int, default=None, help="Value of kc to use.")
    parser.add_argument("--s23_kc", action="store_true", default=False,
                        help="Choose kc with S23's procedure only, without the extended search of "
                             "the paper's Appendix 1.")
    parser.add_argument("--do_sn", action="store_true", default=False,
                        help="Also find the S/N-maximising delta DM (as SHRINE's maximise_sn.py)")
    parser.add_argument("--sn_only", action="store_true", default=False,
                        help="Find only the S/N-maximising delta DM (no structure maximisation); "
                             "for very low S/N bursts")
    return parser.parse_args()


if __name__ == "__main__":
    main()
