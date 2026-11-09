#!/usr/bin/env python3
"""generate_profiles.py and maximise_structure.py in one go, refining the trial DMs until they are fine enough.

Pass 1 uses the trial DMs given (-L, -H, --dDM or --cDM). If the step is coarser than sigma_DM/3, or the fit window
holds fewer than three trial DMs, the profiles are regenerated on a finer grid about the result (step sigma_DM/6,
out to 15 sigma_DM, as in the paper's simulations) and the estimate is repeated, up to --max_passes passes.

Takes the flags of both scripts. Writes the outputs of maximise_structure.py for the last pass, the final
I(DM, t) as generate_profiles.py would, and each earlier pass's I(DM, t) with a _pass<n> suffix.
With --do_sn, also finds the S/N-maximising DM (maximise_sn.py) on the pass-1 trial DMs, which are the widest, and
plots the smoothed profiles at both DMs ({label}_structure_SN_comparison.png).
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_profiles import trial_dms
from maximise_sn import write_outputs as write_sn_outputs
from maximise_structure import estimate, save_plots
from shrine2 import check_grid, plots
from shrine2.snr import maximise_sn
from shrine2.structure import smooth
from shrine2.profiles import crop, make_profiles, zapped_channels
from shrine2.summary import write_summary


def main():
    """Profiles, Section 3.3 and finer trial DMs until they are fine enough; then the outputs."""
    a = get_args()
    X = np.load(f"{a.label}_X_t_{a.DM}.npy", mmap_mode="r")
    Y = np.load(f"{a.label}_Y_t_{a.DM}.npy", mmap_mode="r")
    X, Y = crop(X, Y, np.load(a.I, mmap_mode="r"), a.crop_dur, a.force_peak)

    zap = zapped_channels(a.zap, a.zap_edges)
    d = trial_dms(a, zap)
    for n in range(1, a.max_passes + 1):
        I = make_profiles(X, Y, d, a.f0, a.bw, a.dt, zap)
        if n == 1:
            res, note, kc_warn = estimate(I, d, a, zap)
            a.force_kc = res["kc"]                       # keep k_c from the widest trial DMs in later passes
            I_1, d_1 = I, d
        else:
            res, _, _ = estimate(I, d, a, zap)
        grid = check_grid(res)[1]                        # finer trial DMs, or None if these are fine enough
        msg = f"{res['d_fit']:.6g} +/- {res['sig_DM']:.3g}" if res["ok"] else res["why"]
        print(f"pass {n}: {d.size} trial DMs, step {d[1] - d[0]:.3g}: {msg}")
        if grid is None or n == a.max_passes:
            break
        np.save(f"{a.label}_DMs_pass{n}.npy", d)
        np.save(f"{a.label}_I_{a.dt}us_pass{n}.npy", I)
        lo, hi, step = grid
        d = np.arange(lo, hi + step / 2, step)

    np.save(f"{a.label}_DMs.npy", d)
    np.save(f"{a.label}_I_{a.dt}us.npy", I)
    with open(f"{a.label}_zap.txt", "w") as out:
        out.write(" ".join(str(c) for c in zap) + "\n")
    warnings, grid = check_grid(res)
    warnings = kc_warn + warnings
    np.savetxt(f"{a.label}_SPs.dat", res["Sp"])
    np.save(f"{a.label}_I_smooth_{a.dt}us.npy", smooth(I, res["kc"]))
    write_summary(f"{a.label}_structure_summaryfile.txt", res, a.label, a.dt, a.DM, note, warnings, grid, a.save_all,
                  zap)
    if a.save_all:
        save_plots(res, I, a)
    for x in warnings:
        print("Warning:", x)

    if a.do_sn:
        sn = maximise_sn(I_1, d_1, res["kc"], a.dt)
        write_sn_outputs(sn, I_1, a.label, a.dt, a.save_all)
        i_st = int(np.argmin(np.abs(d_1 - (res["d_fit"] if res["ok"] else res["d_max"]))))
        plots.plot_structure_sn(sn["Sx"], d_1, i_st, sn["j_max"], a.dt, f"{a.label}_structure_SN_comparison.png")
        print(f"S/N-maximising delta DM = {sn['d_sn']:.6g} (S/N {sn['SN_max']:.1f})")


def get_args():
    """The command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    p = ArgumentParser(description="Structure-maximising delta DM and its 1-sigma uncertainty from CRAFT voltages.",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-l", "--label", type=str, help="FRB label")
    p.add_argument("-d", "--DM", type=str, help="Dispersion measure to which the input has been dedispersed in pc/cm3")
    p.add_argument("-L", "--DM_low", type=float, default=-5, help="Lower bound of delta DM range (pass 1)")
    p.add_argument("-H", "--DM_high", type=float, default=5, help="Upper bound of delta DM range (pass 1)")
    p.add_argument("--dDM", type=float, default=0.1, help="Delta DM step size in pc/cm3 (pass 1)")
    p.add_argument("--cDM", type=int, default=0, help="Number of DM steps to divide the range into (pass 1)")
    p.add_argument("--dDM_auto", action="store_true", default=False,
                   help="Set the delta DM step to the one-sample DM step for -t, -f and -b (ignores --dDM and --cDM)")
    p.add_argument("-t", "--dt", type=int, default=1, help="Time resolution in us")
    p.add_argument("-f", "--f0", type=float, required=True, help="Central frequency of observation in MHz")
    p.add_argument("-b", "--bw", type=int, default=336, help="Bandwidth of observation in MHz")
    p.add_argument("--nchan", type=int, default=336, help="Number of coarse channels formed from the voltages")
    p.add_argument("--crop_dur", type=int, default=10, help="Duration of the crop in ms")
    p.add_argument("-I", type=str, help="I profile filename (for doing the crop)")
    p.add_argument("--force_peak", type=int, default=None, help="Force the peak to be at this time step (in ms)")
    p.add_argument("--zap", nargs="+", default=[],
                   help="Coarse channels to leave out: indices and inclusive ranges, e.g. --zap 0 100-110 335 "
                        "(SHRINE always left out 335)")
    p.add_argument("--zap_edges", type=int, default=0,
                   help="Leave out the first and last N channel indices (1 = channels 0 and 335)")
    p.add_argument("-s", "--save_all", action="store_true", default=False, help="Save all plots.")
    p.add_argument("-kc", "--force_kc", type=int, default=None, help="Value of kc to use.")
    p.add_argument("--s23_kc", action="store_true", default=False,
                   help="Choose kc with S23's procedure only, without the extended search of the paper's Appendix 1.")
    p.add_argument("--max_passes", type=int, default=3, help="Largest number of passes over the trial DMs")
    p.add_argument("--do_sn", action="store_true", default=False,
                   help="Also find the S/N-maximising delta DM (as SHRINE's maximise_sn.py)")
    return p.parse_args()


if __name__ == "__main__":
    main()
