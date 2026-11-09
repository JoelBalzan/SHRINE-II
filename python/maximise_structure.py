#!/usr/bin/env python3
"""The structure-maximising delta DM of an FRB and its 1-sigma uncertainty, from I(DM, t).

Same flags, inputs and main outputs as SHRINE's maximise_structure.py, with the uncertainties of
the paper (Section 3.3) in place of S23's.

Inputs  (from generate_profiles.py, in the working directory)
    {label}_DMs.npy, {label}_I_{dt}us.npy
    {label}_zap.txt (if present)         the zapped channels, unless --zap/--zap_edges are given
Outputs
    {label}_structure_summaryfile.txt    k_c, d_fit +- sigma_DM, Sp_max +- sigma_S, diagnostics and
                                         warnings
    {label}_SPs.dat                      the structure parameter at each trial delta-DM
    {label}_I_smooth_{dt}us.npy          the smoothed profiles S x (for structure_SN_comparison.py)
    with -s: DM.txt and plots ({label}_{dt}us_*.png)
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shrine2 import check_grid, choose_kc, dm_uncertainty
from shrine2.profiles import channel_delays, zapped_channels
from shrine2.structure import smooth
from shrine2.summary import write_summary


def zapped(args):
    """The zapped channels: from --zap/--zap_edges if given, otherwise from {label}_zap.txt if it
    exists.

    The channel delays must describe the channels generate_profiles.py actually summed, which it
    records in {label}_zap.txt.

    Args:
        args: the parsed flags; uses --zap, --zap_edges and -l.
    Returns:
        sorted array of channel indices (empty if none).
    """
    if args.zap or args.zap_edges:
        return zapped_channels(args.zap, args.zap_edges)

    path = Path(f"{args.label}_zap.txt")
    if path.exists():
        return np.array(path.read_text().split(), dtype=int)
    return np.array([], dtype=int)


def estimate(I, d, args, zap=()):
    """k_c, then Section 3.3. Returns (result, note on how k_c was chosen, warnings about k_c).

    Args:
        I: I(d_j, t), shape (n_d, N).
        d: the trial DMs, shape (n_d,), pc cm^-3.
        args: the parsed flags; uses -kc, --s23_kc, -f, -b, --nchan and -t.
        zap: the channels left out of the profiles.
    Returns:
        res: the dict returned by estimate.dm_uncertainty.
        note: str, how k_c was chosen.
        warnings: list of str, warnings about k_c.
    """
    # The automatic k_c. If it cannot be found, that only matters when no k_c was forced.
    try:
        kc_auto = choose_kc(I, extend=not args.s23_kc)
    except RuntimeError:
        if args.force_kc is None:
            raise
        kc_auto = None

    if args.s23_kc:
        note = "S23's procedure"
    else:
        note = "S23's procedure, with the extended search (paper Appendix 1)"
    kc = kc_auto
    warnings = []

    if args.force_kc is not None:
        kc = args.force_kc
        note = f"forced (-kc); the automatic choice is {kc_auto}"

        # sigma_t^2 uses modes k > k_min = max(N/4, 2 k_c) (Sec. 3.3.2), assumed to lie above the
        # burst's power. A k_c below the automatic one can put burst power in those modes,
        # inflating sigma_S and sigma_DM.
        if kc_auto is not None and kc < kc_auto:
            warnings.append(f"forced k_c = {kc} is below the automatic {kc_auto}: sigma_t^2 may "
                            "include burst power, which overestimates sigma_S and sigma_DM")

    # Each summed channel's delay (us per pc cm^-3) and share of the noise; dt_nu is in samples.
    tau, share = channel_delays(args.f0, args.bw, args.nchan, zap)
    dt_nu = tau / args.dt

    res = dm_uncertainty(I, d, kc, dt_nu, share)
    return res, note, warnings


def save_plots(res, I, args):
    """DM.txt and the diagnostic plots ({label}_{dt}us_*.png), as SHRINE with -s.

    Args:
        res: the dict returned by estimate.dm_uncertainty.
        I: I(d_j, t), shape (n_d, N).
        args: the parsed flags; uses -l and -t.
    """
    # Imported here so that matplotlib is only loaded when plots are wanted.
    from shrine2 import plots

    tag = f"{args.label}_{args.dt}us"

    # The profile plotted: the trial DM nearest d_fit, or d_max if there was no fit.
    if res["ok"]:
        j = int(np.argmin(np.abs(res["d"] - res["d_fit"])))
    else:
        j = res["j_max"]

    # As SHRINE: DM.txt holds the row of I(DM, t) with the largest structure parameter.
    with open("DM.txt", "w") as out:
        out.write(f"{res['j_max']}")

    plots.plot_structure(res, f"{tag}_structure_parameter.png")
    plots.plot_dct(I, res["kc"], f"{tag}_DCT_v_k_index.png")
    plots.plot_profile(I[j], res["kc"], args.dt, f"{tag}_I_max.png")
    plots.plot_I_DM_t(I, res["d"], args.dt, f"{tag}_I_DM_t.png")


def write_outputs(res, I, args, kc_note, warnings, grid, zap):
    """Everything maximise_structure.py writes for a result of estimate.dm_uncertainty.

    {label}_SPs.dat, {label}_I_smooth_{dt}us.npy, {label}_structure_summaryfile.txt and, with -s,
    DM.txt and the plots.

    Args:
        res: the dict returned by estimate.dm_uncertainty.
        I: I(d_j, t), shape (n_d, N).
        args: the parsed flags; uses -l, -t, -d and -s.
        kc_note: how k_c was chosen.
        warnings: list of str, for the summary file.
        grid: (d_low, d_high, step) from check_grid, or None.
        zap: the channels left out of the profiles.
    """
    np.savetxt(f"{args.label}_SPs.dat", res["Sp"])

    # As SHRINE, for structure_SN_comparison.py.
    np.save(f"{args.label}_I_smooth_{args.dt}us.npy", smooth(I, res["kc"]))

    write_summary(f"{args.label}_structure_summaryfile.txt", res, args.label, args.dt, args.DM,
                  kc_note, warnings, grid, args.save_all, zap)
    if args.save_all:
        save_plots(res, I, args)


def main():
    """Read I(DM, t), run Section 3.3 and write the summary, SPs.dat and (with -s) plots."""
    args = get_args()
    d = np.load(f"{args.label}_DMs.npy")
    I = np.load(f"{args.label}_I_{args.dt}us.npy")
    zap = zapped(args)

    res, kc_note, kc_warnings = estimate(I, d, args, zap)
    grid_warnings, grid = check_grid(res)
    warnings = kc_warnings + grid_warnings

    write_outputs(res, I, args, kc_note, warnings, grid, zap)

    if res["ok"]:
        print(f"delta DM = {res['d_fit']:.6g} +/- {res['sig_DM']:.3g} (1 sigma), k_c = {res['kc']}")
    for message in warnings:
        print("Warning:", message)
    if grid is not None:
        d_low, d_high, step = grid
        print(f"Rerun generate_profiles.py with -L {d_low:.6g} -H {d_high:.6g} --dDM {step:.3g}")


def get_args():
    """The command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    parser = ArgumentParser(
        description="Calculates structure maximising delta DM for pre-dedispersed data.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-l", "--label", type=str, help="FRB label")
    parser.add_argument("-t", "--dt", type=int, default=1, help="Time resolution of inputs in us.")
    parser.add_argument("-d", "--DM", type=float,
                        help="Dispersion measure to which the input has been dedispersed in pc/cm3")
    parser.add_argument("-s", "--save_all", action="store_true", default=False,
                        help="Save all plots.")
    parser.add_argument("-kc", "--force_kc", type=int, default=None, help="Value of kc to use.")
    parser.add_argument("-f", "--f0", type=float, required=True,
                        help="Central frequency of observation in MHz (as given to "
                             "generate_profiles.py); needed for the channel delays in sigma_DM.")
    parser.add_argument("-b", "--bw", type=float, default=336,
                        help="Bandwidth of observation in MHz (as given to generate_profiles.py).")
    parser.add_argument("--nchan", type=int, default=336,
                        help="Number of coarse channels formed by generate_profiles.py.")
    parser.add_argument("--s23_kc", action="store_true", default=False,
                        help="Choose kc with S23's procedure only, without the extended search of "
                             "the paper's Appendix 1.")
    parser.add_argument("--zap", nargs="+", default=[],
                        help="Coarse channels to leave out: indices and inclusive ranges, e.g. "
                             "--zap 0 100-110 335 (default: those in {label}_zap.txt from "
                             "generate_profiles.py)")
    parser.add_argument("--zap_edges", type=int, default=0,
                        help="Leave out the first and last N channel indices (1 = channels 0 and 335)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
