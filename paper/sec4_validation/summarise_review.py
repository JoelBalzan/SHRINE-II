"""Numbers for the Monte Carlo tests (run_mc_review.py): writes results/review_numbers.json.

For each test and evaluation: the Monte Carlo scatter of d_fit about d_0 (sigma_MC), the median
quoted sigma_DM over sigma_MC (q), the coverage of d_fit +- sigma_DM and +- 2 sigma_DM, the standard
deviation of (d_fit - d_0)/sigma_DM (pull), the bias in units of sigma_MC, the median quoted
sigma_S over the fractional scatter of Sp at the trial DM nearest d_0, and the 16th-84th
percentiles of sigma_DM over its median.

usage: python summarise_review.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sims  # noqa: E402


def stats(d_fit, sig_DM, sig_S, Sp_d0, d0):
    ok = np.isfinite(d_fit) & np.isfinite(sig_DM)
    dd = d_fit[ok] - d0
    s = sig_DM[ok]
    n = ok.sum()
    sig_mc = np.std(dd)
    med = np.median(s)
    z = dd / s
    return dict(
        n=int(n), n_failed=int((~ok).sum()), sig_mc=float(sig_mc), q=float(med / sig_mc),
        q_err=float(1 / np.sqrt(2 * (n - 1))), cover1=float(np.mean(np.abs(z) < 1)),
        cover2=float(np.mean(np.abs(z) < 2)), pull=float(np.std(z)), bias=float(np.mean(dd) / sig_mc),
        p16=float(np.percentile(s, 16) / med), p84=float(np.percentile(s, 84) / med),
        sp_q=float(np.median(sig_S) / np.std(Sp_d0)))


def main():
    out = {}
    for path in sorted(glob.glob(sims.results_path("review_*.npz"))):
        if os.path.basename(path).startswith("review_selfnoise"):
            continue  # the old self-noise runs; superseded by run_mc_selfnoise.py
        f = np.load(path)
        meta = json.loads(str(f["meta"]))
        key = os.path.basename(path)[7:-4]
        models = sorted({k[len("d_fit_"):] for k in f.files if k.startswith("d_fit_")})
        row = dict(meta=meta)
        for m in models:
            row[m] = stats(f[f"d_fit_{m}"], f[f"sig_DM_{m}"], f[f"sig_S_{m}"], f["Sp_d0"],
                           meta["d0"])
            row[m]["c2_err_median"] = float(np.nanmedian(f[f"c2_err_{m}"]))
        for k in ["var_ratio"]:
            if k in f.files:
                v = f[k]
                row[k] = dict(median=float(np.nanmedian(v)), p16=float(np.nanpercentile(v, 16)),
                              p84=float(np.nanpercentile(v, 84)))
        out[key] = row
        line = ", ".join(f"{m}: q {row[m]['q']:.3f}, cover {row[m]['cover1']:.3f}/"
                         f"{row[m]['cover2']:.3f}, pull {row[m]['pull']:.3f}, bias "
                         f"{row[m]['bias']:+.3f}, sigS/MC {row[m]['sp_q']:.3f}" for m in models)
        extra = {k: round(row[k]["median"], 3) for k in ["var_ratio"] if k in row}
        print(f"{key} (n={meta['n_real']}): {line} {extra}")
    json.dump(out, open(sims.results_path("review_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
