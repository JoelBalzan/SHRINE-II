"""The numbers of Section 4 (and those of Appendix A, the abstract and the conclusions that come
from the same runs), from the Monte Carlo results of run_mc.py, run_mc_kc.py and run_window.py.

Per burst and S/N, with sigma_MC the standard deviation of d_fit over the realisations whose fit
succeeded:
    q            median quoted sigma_DM / sigma_MC (Figure 7c), with its sampling error
                 1 / sqrt(2 (n - 1));
    p16, p84     16th and 84th percentiles of the quoted sigma_DM / its median;
    cover        fraction of realisations with |d_fit - d_0| <= sigma_DM;
    bias         (mean d_fit - d_0) / sigma_MC;
    half_over_mc median half-width of the fit window / sigma_MC; n_win_min, n_win_max its size;
    s23_over_mc  median half-width of S23's interval / sigma_MC (Figure 7c), and the fraction of
                 S23 intervals that reach the end of their trial DMs (lower limits);
    cross_over_mc median half-width of the sigma_S crossing range / sigma_MC (Figure 7c);
    sp_new, sp_s23 median sigma_S / Sp_max and S23's relative uncertainty, each divided by the
                 fractional scatter of Sp at the trial DM nearest d_0 (Figure 7d);
    second       sigma_MC / the scatter of d_par (the vertex without the second-order term);
    c2_err_median, c2_err_frac_gt_half: the relative curvature error std[c2]/|c2|;
    d0, d0_over_sigma: d_0 relative to the true DM, in pc cm^-3 and in units of the median sigma_DM;
from run_mc_kc.py:
    kc_ratio_var, kc_ratio_ext: the scatter of d_fit with k_c chosen per realisation (S23's
                 procedure; with the extended search) / that with k_c fixed;
    kc_q_var, kc_q_ext: median quoted sigma_DM / that scatter;
    kc_q_var_d0, kc_q_ext_d0: the same with the scatter measured about d_0(k_c);
and from run_window.py (S/N = 15, 40, 200), for each window half-width k sigma_1 narrower than the
sigma_S window: the median quoted sigma_DM / scatter, and the scatter / that of the sigma_S-window
fit. Refits whose vertex falls outside their own window are left out.

Writes results/section4_numbers.json: the rows, and "paper", the ranges as the paper quotes them.

usage: python summarise.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402


def case_numbers(tag, snr):
    """The numbers for one burst and S/N from run_mc.py and run_mc_kc.py."""
    mc = np.load(sims.results_path(f"mc_{tag}_snr{snr}.npz"))
    meta = json.loads(str(mc["meta"]))
    d0 = meta["d0"]
    ok = np.isfinite(mc["d_fit"]) & np.isfinite(mc["sig_DM"])
    n = int(ok.sum())
    d_fit, sig_DM = mc["d_fit"][ok], mc["sig_DM"][ok]
    sig_mc = float(np.std(d_fit))
    median = float(np.median(sig_DM))

    # Sp: the fractional scatter at the trial DM nearest d_0.
    Sp_frac_scatter = np.std(mc["Sp_d0"]) / np.mean(mc["Sp_d0"])

    # S23's interval was found for the first realisations only.
    has_s23 = np.isfinite(mc["s23_lo"])
    s23_half = 0.5 * (mc["s23_hi"][has_s23] - mc["s23_lo"][has_s23])

    half = 0.5 * (mc["d_hi"][ok] - mc["d_lo"][ok])
    both = ok & np.isfinite(mc["d_par"])
    c2_err = mc["c2_err"][ok]

    row = dict(tag=tag, snr=snr, n=n, n_failed=int((~ok).sum()), kc=meta["kc"], d0=d0,
               sig_mc=sig_mc, sig_DM_median=median, sig_DM_pred=meta["sig_DM_pred"],
               q=median / sig_mc, q_err=1 / np.sqrt(2 * (n - 1)),
               p16=float(np.percentile(sig_DM, 16) / median), p84=float(np.percentile(sig_DM, 84) / median),
               cover=float(np.mean(np.abs(d_fit - d0) <= sig_DM)),
               bias=float((np.mean(d_fit) - d0) / sig_mc),
               half_over_mc=float(np.median(half) / sig_mc),
               n_win_min=int(np.min(mc["n_win"][ok])), n_win_max=int(np.max(mc["n_win"][ok])),
               s23_over_mc=float(np.median(s23_half) / sig_mc),
               s23_at_edge=float(np.mean(mc["s23_edge"][has_s23])),
               cross_over_mc=float(np.nanmedian(mc["h_cross"]) / sig_mc),
               sp_new=float(np.median(mc["sig_S"] / mc["Sp_max"]) / Sp_frac_scatter),
               sp_s23=float(np.median(mc["s23_rel"]) / Sp_frac_scatter),
               sp_err=1 / np.sqrt(2 * (mc["Sp_d0"].size - 1)),
               second=float(np.std(mc["d_fit"][both]) / np.std(mc["d_par"][both])),
               c2_err_median=float(np.median(c2_err)),
               c2_err_frac_gt_half=float(np.mean(c2_err > 0.5)),
               d0_over_sigma=d0 / median)

    # k_c chosen per realisation.
    kc_path = sims.results_path(f"kc_{tag}_snr{snr}.npz")
    if os.path.exists(kc_path):
        kc = np.load(kc_path)
        ok_kc = np.isfinite(kc["d_fix"]) & np.isfinite(kc["d_var"]) & np.isfinite(kc["d_ext"])
        s_fix = np.std(kc["d_fix"][ok_kc])
        s_var = np.std(kc["d_var"][ok_kc])
        s_ext = np.std(kc["d_ext"][ok_kc])
        row.update(kc_n=int(ok_kc.sum()), kc_ratio_var=float(s_var / s_fix), kc_ratio_ext=float(s_ext / s_fix),
                   kc_q_var=float(np.median(kc["s_var"][ok_kc]) / s_var),
                   kc_q_ext=float(np.median(kc["s_ext"][ok_kc]) / s_ext),
                   kc_q_var_d0=float(np.median(kc["s_var"][ok_kc])
                                     / np.std(kc["d_var"][ok_kc] - kc["d0_var"][ok_kc])),
                   kc_q_ext_d0=float(np.median(kc["s_ext"][ok_kc])
                                     / np.std(kc["d_ext"][ok_kc] - kc["d0_ext"][ok_kc])),
                   kc_frac_changed_var=float(np.mean(kc["kc_var"][ok_kc] != meta["kc"])),
                   kc_frac_changed_ext=float(np.mean(kc["kc_ext"][ok_kc] != kc["kc_var"][ok_kc])))
    return row, mc


def window_numbers(tag, snr):
    """The fit-window test for one burst and S/N from run_window.py."""
    w = np.load(sims.results_path(f"window_{tag}_snr{snr}.npz"))
    meta = json.loads(str(w["meta"]))
    ks = np.array(meta["ks"])
    ok1 = np.isfinite(w["d1"]) & np.isfinite(w["s1"])
    sd1 = np.std(w["d1"][ok1])
    row = dict(tag=tag, snr=snr, n=int(ok1.sum()), ks=ks.tolist(),
               q_window=float(np.median(w["s1"][ok1]) / sd1),
               half_over_sigma=float(np.median(w["h1"][ok1] / w["s1"][ok1])), q=[], sd_rel=[], n_k=[])
    for i in range(ks.size):
        ok = np.isfinite(w["d_fit"][:, i]) & np.isfinite(w["sig_DM"][:, i])
        ok &= np.abs(w["d_fit"][:, i] - w["d1"]) <= ks[i] * w["s1"]  # vertex inside its window
        sd = np.std(w["d_fit"][ok, i])
        row["q"].append(float(np.median(w["sig_DM"][ok, i]) / sd))
        row["sd_rel"].append(float(sd / sd1))
        row["n_k"].append(int(ok.sum()))
    return row


def value_range(rows, *keys):
    """[min, max] of row[key] over the rows and keys given."""
    values = []
    for row in rows:
        for key in keys:
            values.append(row[key])
    return [float(min(values)), float(max(values))]


def main():
    rows = []
    largest = []  # (sig_DM / median, c2_err, tag, snr, realisation) for every realisation
    for tag in ["single", "double"]:
        for snr in sims.SNRS:
            row, mc = case_numbers(tag, snr)
            rows.append(row)
            print(f"{tag:6s} S/N {snr:3d} n {row['n']:4d} | q {row['q']:.3f}+-{row['q_err']:.3f} "
                  f"(p16/84 {row['p16']:.2f}/{row['p84']:.2f}) cover {row['cover']:.3f} bias {row['bias']:+.3f} | "
                  f"window {row['half_over_mc']:.1f} sigma_MC, {row['n_win_min']}-{row['n_win_max']} trial DMs | "
                  f"S23 {row['s23_over_mc']:.1f} cross {row['cross_over_mc']:.1f} | Sp {row['sp_new']:.3f} "
                  f"S23 {row['sp_s23']:.2f} | second {row['second']:.3f} c2_err {row['c2_err_median']:.3f} "
                  f"(>0.5: {row['c2_err_frac_gt_half']:.3f}) | d0 {row['d0']:.4f} = {row['d0_over_sigma']:.1f} sigma_DM")
            if "kc_ratio_var" in row:
                print(f"{'':17s} k_c per realisation: scatter x{row['kc_ratio_var']:.3f} (extended "
                      f"x{row['kc_ratio_ext']:.3f}); q {row['kc_q_var']:.3f} ({row['kc_q_ext']:.3f}), "
                      f"about d0(k_c) {row['kc_q_var_d0']:.3f} ({row['kc_q_ext_d0']:.3f})")
            ok = np.isfinite(mc["sig_DM"])
            median = np.median(mc["sig_DM"][ok])
            for r in np.flatnonzero(ok):
                largest.append((mc["sig_DM"][r] / median, mc["c2_err"][r], tag, snr, int(r)))
    largest.sort(reverse=True)

    windows = []
    for tag in ["single", "double"]:
        for snr in [15, 40, 200]:
            if os.path.exists(sims.results_path(f"window_{tag}_snr{snr}.npz")):
                w = window_numbers(tag, snr)
                windows.append(w)
                line = (f"window {tag:6s} S/N {snr:3d}: sigma_S window q {w['q_window']:.3f} "
                        f"(half-width {w['half_over_sigma']:.1f} sigma_DM) | k: q/scatter")
                for k, q, sd in zip(w["ks"], w["q"], w["sd_rel"]):
                    line += f" {k:g}: {q:.3f}/{sd:.3f}"
                print(line)

    # The ranges as the paper quotes them, over all cases or some.
    q_dev = []
    sp_dev = []
    rows_15, rows_200, rows_2c40, rows_not_2c40, rows_c2 = [], [], [], [], []
    for row in rows:
        q_dev.append(abs(row["q"] - 1))
        sp_dev.append(abs(row["sp_new"] - 1))
        if row["snr"] == 15:
            rows_15.append(row)
        if row["snr"] == 200:
            rows_200.append(row)
        if row["tag"] == "double" and row["snr"] == 40:
            rows_2c40.append(row)
        else:
            rows_not_2c40.append(row)
        if row["c2_err_frac_gt_half"] > 0:
            rows_c2.append(row["snr"])
    paper = dict(
        sigma_DM_max_deviation=float(max(q_dev)),
        sigma_S_max_deviation=float(max(sp_dev)),
        s23_dm_factor=value_range(rows, "s23_over_mc"),
        s23_Sp_factor=value_range(rows, "sp_s23"),
        crossing_factor=value_range(rows, "cross_over_mc"),
        p16_p84_snr15=[value_range(rows_15, "p16"), value_range(rows_15, "p84")],
        p16_p84_snr200=[value_range(rows_200, "p16"), value_range(rows_200, "p84")],
        second_order=value_range(rows, "second"),
        window_half_over_mc=value_range(rows, "half_over_mc"),
        window_n_trial_dms=value_range(rows, "n_win_min", "n_win_max"),
        coverage=value_range(rows, "cover"),
        max_abs_bias=float(max(np.abs(value_range(rows, "bias")))),
        c2_err_median=value_range(rows, "c2_err_median"),
        c2_err_frac_gt_half_max=value_range(rows, "c2_err_frac_gt_half")[1],
        c2_err_gt_half_snrs=sorted(set(rows_c2)),
        c2_err_of_largest_sigma_DM=[],
        d0=value_range(rows, "d0"),
        d0_over_sigma_DM=value_range(rows, "d0_over_sigma"),
    )
    # The two realisations with the largest quoted sigma_DM (relative to their case's median).
    for x in largest[:2]:
        paper["c2_err_of_largest_sigma_DM"].append(
            dict(sig_over_median=float(x[0]), c2_err=float(x[1]), tag=x[2], snr=x[3], realisation=x[4]))
    have_kc = True
    for row in rows:
        if "kc_ratio_var" not in row:
            have_kc = False
    if have_kc:
        paper.update(kc_scatter_increase_except_2c40=value_range(rows_not_2c40, "kc_ratio_var"),
                     kc_scatter_increase_2c40=rows_2c40[0]["kc_ratio_var"],
                     kc_scatter_increase_2c40_extended=rows_2c40[0]["kc_ratio_ext"],
                     kc_scatter_increase_extended=value_range(rows, "kc_ratio_ext"),
                     kc_q_about_d0_kc=value_range(rows, "kc_q_var_d0", "kc_q_ext_d0"),
                     kc_q_extended=value_range(rows, "kc_q_ext"))
    if windows:
        q_change = []
        sd_change = []
        for w in windows:
            for k, q, sd in zip(w["ks"], w["q"], w["sd_rel"]):
                if k < w["half_over_sigma"]:
                    q_change.append(abs(q - w["q_window"]))
                    sd_change.append(abs(sd - 1))
        paper.update(narrower_windows_max_change_q=float(max(q_change)),
                     narrower_windows_max_change_scatter=float(max(sd_change)))
    print(json.dumps(paper, indent=1))

    json.dump(dict(rows=rows, windows=windows, paper=paper),
              open(sims.results_path("section4_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
