#!/usr/bin/env python3
"""The structure-maximising delta DM of an FRB and its 1-sigma uncertainty, from I(DM, t).

Same flags, inputs and main outputs as SHRINE's maximise_structure.py, with the uncertainties of the paper
(Section 3.3) in place of S23's.

Inputs  (from generate_profiles.py, in the working directory)
    {label}_DMs.npy, {label}_I_{dt}us.npy
Outputs
    {label}_structure_summaryfile.txt    k_c, d_fit +- sigma_DM, Sp_max +- sigma_S, diagnostics and warnings
    {label}_SPs.dat                      the structure parameter at each trial delta-DM
    {label}_I_smooth_{dt}us.npy          the smoothed profiles S x (for structure_SN_comparison.py)
    with -s: DM.txt and plots ({label}_{dt}us_*.png)
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run from a checkout without installing
from shrine2 import check_grid, choose_kc, dm_uncertainty
from shrine2.profiles import channel_delays, zapped_channels
from shrine2.structure import smooth
from shrine2.summary import write_summary


def zapped(a):
    """The zapped channels: from --zap/--zap_edges if given, otherwise from {label}_zap.txt if it exists.

    The channel delays must describe the channels generate_profiles.py actually summed, which it records in
    {label}_zap.txt.

    Args:
        a: the parsed flags; uses --zap, --zap_edges and -l.
    Returns:
        sorted array of channel indices (empty if none).
    """
    if a.zap or a.zap_edges:
        return zapped_channels(a.zap, a.zap_edges)
    path = Path(f"{a.label}_zap.txt")
    return np.array(path.read_text().split(), dtype=int) if path.exists() else np.array([], dtype=int)


def estimate(I, d, a, zap=()):
    """k_c, then Section 3.3. Returns (result, note on how k_c was chosen, warnings about k_c).

    Args:
        I: I(d_j, t), shape (n_d, N).
        d: the trial DMs, shape (n_d,), pc cm^-3.
        a: the parsed flags; uses -kc, --s23_kc, -f, -b, --nchan and -t.
        zap: the channels left out of the profiles.
    Returns:
        res: the dict returned by estimate.dm_uncertainty.
        note: str, how k_c was chosen.
        warn: list of str, warnings about k_c.
    """
    try:
        kc_auto = choose_kc(I, extend=not a.s23_kc)
    except RuntimeError:
        if a.force_kc is None:
            raise
        kc_auto = None                                   # no automatic k_c, but one was given
    note = "S23's procedure" + ("" if a.s23_kc else ", with the extended search (paper Appendix 1)")
    kc, warn = kc_auto, []
    if a.force_kc is not None:
        kc, note = a.force_kc, f"forced (-kc); the automatic choice is {kc_auto}"
        if kc_auto is not None and kc < kc_auto:
            # sigma_t^2 uses modes k > k_min = max(N/4, 2 k_c) (Sec. 3.3.2), assumed to lie above the burst's power.
            # A k_c below the automatic one can put burst power in those modes, inflating sigma_S and sigma_DM.
            warn.append(f"forced k_c = {kc} is below the automatic {kc_auto}: sigma_t^2 may include burst power, "
                        "which overestimates sigma_S and sigma_DM")
    tau, share = channel_delays(a.f0, a.bw, a.nchan, zap)   # us per pc cm^-3, and each channel's share of the noise
    return dm_uncertainty(I, d, kc, tau / a.dt, share), note, warn


def save_plots(res, I, a):
    """DM.txt and the diagnostic plots ({label}_{dt}us_*.png), as SHRINE with -s.

    Args:
        res: the dict returned by estimate.dm_uncertainty.
        I: I(d_j, t), shape (n_d, N).
        a: the parsed flags; uses -l and -t.
    """
    from shrine2 import plots
    tag = f"{a.label}_{a.dt}us"
    j = int(np.argmin(np.abs(res["d"] - res["d_fit"]))) if res["ok"] else res["j_max"]
    with open("DM.txt", "w") as out:                    # as SHRINE: the row of I(DM, t) at the maximum
        out.write(f"{res['j_max']}")
    plots.plot_structure(res, f"{tag}_structure_parameter.png")
    plots.plot_dct(I, res["kc"], f"{tag}_DCT_v_k_index.png")
    plots.plot_profile(I[j], res["kc"], a.dt, f"{tag}_I_max.png")
    plots.plot_I_DM_t(I, res["d"], a.dt, f"{tag}_I_DM_t.png")


def main():
    """Read I(DM, t), run Section 3.3 and write the summary, SPs.dat and (with -s) plots."""
    a = get_args()
    d = np.load(f"{a.label}_DMs.npy")
    I = np.load(f"{a.label}_I_{a.dt}us.npy")
    zap = zapped(a)
    res, note, kc_warn = estimate(I, d, a, zap)
    warnings, grid = check_grid(res)
    warnings = kc_warn + warnings

    np.savetxt(f"{a.label}_SPs.dat", res["Sp"])
    np.save(f"{a.label}_I_smooth_{a.dt}us.npy", smooth(I, res["kc"]))      # as SHRINE, for structure_SN_comparison.py
    write_summary(f"{a.label}_structure_summaryfile.txt", res, a.label, a.dt, a.DM, note, warnings, grid, a.save_all,
                  zap)
    if a.save_all:
        save_plots(res, I, a)

    if res["ok"]:
        print(f"delta DM = {res['d_fit']:.6g} +/- {res['sig_DM']:.3g} (1 sigma), k_c = {res['kc']}")
    for x in warnings:
        print("Warning:", x)
    if grid is not None:
        print(f"Rerun generate_profiles.py with -L {grid[0]:.6g} -H {grid[1]:.6g} --dDM {grid[2]:.3g}")


def get_args():
    """The command-line flags.

    Returns:
        the parsed flags (argparse.Namespace).
    """
    p = ArgumentParser(description="Calculates structure maximising delta DM for pre-dedispersed data.",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-l", "--label", type=str, help="FRB label")
    p.add_argument("-t", "--dt", type=int, default=1, help="Time resolution of inputs in us.")
    p.add_argument("-d", "--DM", type=float, help="Dispersion measure to which the input has been dedispersed in pc/cm3")
    p.add_argument("-s", "--save_all", action="store_true", default=False, help="Save all plots.")
    p.add_argument("-kc", "--force_kc", type=int, default=None, help="Value of kc to use.")
    p.add_argument("-f", "--f0", type=float, required=True,
                   help="Central frequency of observation in MHz (as given to generate_profiles.py); "
                        "needed for the channel delays in sigma_DM.")
    p.add_argument("-b", "--bw", type=float, default=336, help="Bandwidth of observation in MHz (as given to generate_profiles.py).")
    p.add_argument("--nchan", type=int, default=336, help="Number of coarse channels formed by generate_profiles.py.")
    p.add_argument("--s23_kc", action="store_true", default=False,
                   help="Choose kc with S23's procedure only, without the extended search of the paper's Appendix 1.")
    p.add_argument("--zap", nargs="+", default=[],
                   help="Coarse channels to leave out: indices and inclusive ranges, e.g. --zap 0 100-110 335 "
                        "(default: those in {label}_zap.txt from generate_profiles.py)")
    p.add_argument("--zap_edges", type=int, default=0,
                   help="Leave out the first and last N channel indices (1 = channels 0 and 335)")
    return p.parse_args()


if __name__ == "__main__":
    main()
