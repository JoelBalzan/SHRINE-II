#!/usr/bin/env python3
"""generate_profiles.py and maximise_structure.py in one go, refining the trial DMs until they are
fine enough.

Pass 1 uses the trial DMs given (-L, -H, --dDM or --cDM). If the step is coarser than sigma_DM/3,
or the fit window holds fewer than three trial DMs, the profiles are regenerated on a finer grid
about the result (step sigma_DM/6, out to 15 sigma_DM, as in the paper's simulations) and the
estimate is repeated, up to --max_passes passes.

Takes the flags of both scripts. Writes the outputs of maximise_structure.py for the last pass,
the final I(DM, t) as generate_profiles.py would, and each earlier pass's I(DM, t) with a
_pass<n> suffix. With --do_sn, also finds the S/N-maximising DM (maximise_sn.py) on the pass-1
trial DMs, which are the widest.

With --sn_only, finds only the S/N-maximising DM, on the trial DMs given (no structure
maximisation and no refinement): the standalone S/N mode, for very low S/N bursts.
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
from maximise_sn import print_result as print_sn_result
from maximise_sn import write_outputs as write_sn_outputs
from maximise_structure import choose_kc_from_flags, estimate, low_sn_warnings
from maximise_structure import write_outputs as write_structure_outputs
from shrine2 import check_grid
from shrine2.profiles import crop, make_profiles, zapped_channels
from shrine2.snr import maximise_sn


def sn_only(X, Y, d, zap, args):
    """--sn_only: the profiles on the trial DMs given, then S/N maximisation alone.

    Writes the outputs of generate_profiles.py (except its summary file) and of maximise_sn.py.

    Args:
        X, Y: cropped complex voltages, shape (n,).
        d: the trial DMs, shape (n_d,), pc cm^-3.
        zap: the channels left out of the profiles.
        args: the parsed flags.
    """
    I = make_profiles(X, Y, d, args.f0, args.bw, args.dt, zap)
    np.save(f"{args.label}_DMs.npy", d)
    np.save(f"{args.label}_I_{args.dt}us.npy", I)
    write_zap_file(args.label, zap)

    kc = kc_for_sn(I, args)
    sn = maximise_sn(I, d, kc, args.dt)
    write_sn_outputs(sn, I, args.label, args.dt, args.save_all)
    print_sn_result(sn)


def main():
    """Profiles, Section 3.3 and finer trial DMs until they are fine enough; then the outputs."""
    args = get_args()

    # The voltages, cropped about the burst (as generate_profiles.py).
    X = np.load(f"{args.label}_X_t_{args.DM}.npy", mmap_mode="r")
    Y = np.load(f"{args.label}_Y_t_{args.DM}.npy", mmap_mode="r")
    I_for_crop = np.load(args.I, mmap_mode="r")
    X, Y = crop(X, Y, I_for_crop, args.crop_dur, args.force_peak)

    zap = zapped_channels(args.zap, args.zap_edges)
    d = trial_dms(args, zap)

    if args.sn_only:
        sn_only(X, Y, d, zap, args)
        return

    for n in range(1, args.max_passes + 1):
        I = make_profiles(X, Y, d, args.f0, args.bw, args.dt, zap)
        print(f"pass {n}: {d.size} trial DMs, step {d[1] - d[0]:.3g}, {I.shape[1]} samples of "
              f"{args.dt} us")

        # k_c is chosen once, from pass 1, whose trial DMs are the widest, and kept in later passes.
        if n == 1:
            kc, kc_note, kc_warnings = choose_kc_from_flags(I, args, verbose=True)

            # Pass 1's profiles, for the S/N maximisation (--do_sn).
            I_pass1 = I
            d_pass1 = d
        else:
            print(f"  k_c = {kc} (from pass 1)")

        res = estimate(I, d, kc, args, zap, verbose=True)

        # A finer grid of trial DMs, or None if these are fine enough.
        _, grid = check_grid(res)

        if res["ok"]:
            result_text = f"{res['d_fit']:.6g} +/- {res['sig_DM']:.3g}"
        else:
            result_text = res["why"]
        print(f"pass {n} result: {result_text}")

        if grid is None or n == args.max_passes:
            break

        # Keep this pass's trial DMs and profiles, and go round again on the finer grid.
        np.save(f"{args.label}_DMs_pass{n}.npy", d)
        np.save(f"{args.label}_I_{args.dt}us_pass{n}.npy", I)
        d_low, d_high, step = grid
        d = np.arange(d_low, d_high + step / 2, step)

    # The last pass's profiles, as generate_profiles.py writes them.
    np.save(f"{args.label}_DMs.npy", d)
    np.save(f"{args.label}_I_{args.dt}us.npy", I)
    write_zap_file(args.label, zap)

    # The last pass's result, as maximise_structure.py writes it (with pass 1's note on k_c).
    grid_warnings, grid = check_grid(res)
    warnings = kc_warnings + grid_warnings + low_sn_warnings(res)
    write_structure_outputs(res, I, args, kc_note, warnings, grid, zap)
    for message in warnings:
        print("Warning:", message)

    if args.do_sn:
        sn = maximise_sn(I_pass1, d_pass1, res["kc"], args.dt)
        write_sn_outputs(sn, I_pass1, args.label, args.dt, args.save_all)
        print_sn_result(sn)


def get_args():
    """The command-line flags.

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
                        help="Lower bound of delta DM range (pass 1)")
    parser.add_argument("-H", "--DM_high", type=float, default=5,
                        help="Upper bound of delta DM range (pass 1)")
    parser.add_argument("--dDM", type=float, default=0.1,
                        help="Delta DM step size in pc/cm3 (pass 1)")
    parser.add_argument("--cDM", type=int, default=0,
                        help="Number of DM steps to divide the range into (pass 1)")
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
    parser.add_argument("--max_passes", type=int, default=3,
                        help="Largest number of passes over the trial DMs")
    parser.add_argument("--do_sn", action="store_true", default=False,
                        help="Also find the S/N-maximising delta DM (as SHRINE's maximise_sn.py), "
                             "on the pass-1 trial DMs")
    parser.add_argument("--sn_only", action="store_true", default=False,
                        help="Find only the S/N-maximising delta DM, on the trial DMs given (no "
                             "structure maximisation or refinement); for very low S/N bursts")
    args = parser.parse_args()

    if args.max_passes < 1:
        parser.error("--max_passes must be at least 1")
    return args


if __name__ == "__main__":
    main()
