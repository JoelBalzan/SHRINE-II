"""The summary file written by maximise_structure.py and optimise_dm.py (same layout and key lines as SHRINE's)."""


def write_summary(path, res, label, dt, DM, kc_note, warnings, grid, saving=False, zap=()):
    """Write {label}_structure_summaryfile.txt for a result of estimate.dm_uncertainty.

    Args:
        path: output file name.
        res: the dict returned by estimate.dm_uncertainty.
        label: FRB label (-l).
        dt: time resolution, us (-t).
        DM: the DM the input was dedispersed to (-d).
        kc_note: how k_c was chosen.
        warnings: list of str, from check_grid and the k_c check.
        grid: (d_low, d_high, step) from check_grid, or None.
        saving: whether plots were saved (-s).
        zap: the channels left out of the profiles.
    """
    with open(path, "w") as out:
        w = out.write
        w("//begin maximise_structure summary//\n/*\n")
        w(f"FRB Label: {label}\t//should match params.label\n")
        w(f"Time Resolution: {dt}us\t//should match params.timescale\n")
        w(f"Saving: {saving}\t//should match params.saving\n")
        w(f"kc: {res['kc']}\t//{kc_note}\n")
        w(f"Initial DM: {DM}\n")
        w(f"Zapped channels: {' '.join(str(c) for c in zap) or 'none'}\n")
        if res["ok"]:
            w(f"Structure Maximising Delta DM: {res['d_fit']}\n")
            w(f"Uncertainty in Structure Maximising Delta DM: +/-{res['sig_DM']}\t//1 sigma\n")
            w(f"\tVertex of a parabola fitted over delta DM {res['d_lo']} to {res['d_hi']} ({res['n_win']} trial DMs).\n")
            w(f"\tDelta DM of the largest structure parameter on the grid: {res['d_max']}\n")
            w(f"\tRelative curvature error std[c2]/|c2|: {res['c2_err']:.3f}\t//sigma_DM is unreliable as it approaches 1\n")
        else:
            w(f"Structure Maximising Delta DM: {res['d_max']}\t//largest structure parameter on the grid; no fit\n")
            w("Uncertainty in Structure Maximising Delta DM: unknown\n")
        w(f"Structure Parameter at Maximum: {res['Sp_max']} +/- {res['sig_S']}\t//1 sigma\n")
        w(f"\tFiltered noise / structure parameter: sqrt(<||e||^2>)/S = {res['e_ratio']:.3f}\t//first order holds for <~ 0.3\n")
        for x in warnings:
            w(f"\tWarning: {x}.\n")
        if grid is not None:
            lo, hi, step = grid
            w(f"\tSuggested trial DMs for generate_profiles.py: -L {lo:.6g} -H {hi:.6g} --dDM {step:.3g}\n")
        w("*/\n//end maximise_structure summary//\n\n\n")


def write_sn_summary(path, sn, dt):
    """Write {label}_SN_summaryfile.txt for a result of snr.maximise_sn (SHRINE's maximise_sn layout).

    Args:
        path: output file name.
        sn: the dict returned by snr.maximise_sn.
        dt: time resolution, us (-t).
    """
    d, j, lo, hi = sn["d"], sn["j_max"], sn["lo"], sn["hi"]
    t0, t1 = sn["start"], sn["start"] + sn["length"]
    with open(path, "w") as out:
        w = out.write
        w("//begin maximise_sn summary//\n/*\n")
        w(f"Max SNR at index {j}\n")
        w(f"Uncertainty bounds at indexes {lo} - {hi}\n")
        w(f"Corresponds to a delta DM of {d[j]} +{d[hi] - d[j]} -{d[j] - d[lo]}\n")
        if not (sn["lo_found"] and sn["hi_found"]):
            w("\tThe S/N does not fall by 1 within the trial DMs on that side: the bound is a lower limit.\n")
        w(f"Calculated with window from {t0} to {t1}\n")
        w(f"Corresponds to a time window from {t0 * dt / 1000}ms to {t1 * dt / 1000}ms\n")
        w(f"Max SNR was {sn['SN_max']}\n")
        w(f"\t//noise sigma = std of I - S I over all trial DMs; k_c = {sn['kc']}\n")
        w("*/\n//end maximise_sn summary//\n\n\n")
