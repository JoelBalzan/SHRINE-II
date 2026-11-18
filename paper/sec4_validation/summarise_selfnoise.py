"""The Monte Carlo with noise whose variance follows the burst (run_mc_selfnoise.py): the numbers
of Section 4.2.

For each case and each way of evaluating sigma_t^2 (est: estimated from the profile, as for real
data; true: the true sigma_t^2 at d_max):
    q        median quoted sigma_DM / the standard deviation of d_fit, with the sampling error
             1 / sqrt(2 (n - 1));
    sp       median sigma_S / Sp_max over the fractional scatter of Sp at the trial DM nearest d_0;
    cover    fraction of realisations with |d_fit - d_0| <= sigma_DM;
    pull     standard deviation of (d_fit - d_0) / sigma_DM;
    n_failed realisations whose window or fit failed;
and per case
    pred_over_white  the predicted sigma_DM of the noiseless burst over that for white noise of
             the same S/N (sims.pilot);
    sig2_peak, sig2_high  the mean over realisations of the estimated sigma_t^2 over the true one
             at d_max, at the true variance's peak and over the samples where it exceeds 1.5 times
             its off-pulse value.

Writes results/selfnoise_numbers.json: the rows, and "paper", the ranges over the cases.

usage: python summarise_selfnoise.py   (after run_selfnoise.sh)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402

CASES = [("single", 15), ("single", 40), ("single", 200), ("double", 40)]


def main():
    rows = []
    for tag, snr in CASES:
        path = sims.results_path(f"selfnoise_{tag}_snr{snr}.npz")
        if not os.path.exists(path):
            continue
        a = np.load(path)
        meta = json.loads(str(a["meta"]))
        d0 = meta["d0"]
        Sp_scatter = np.std(a["Sp_d0"]) / np.mean(a["Sp_d0"])
        row = dict(tag=tag, snr=snr, n_real=meta["n_real"], kc=meta["kc"],
                   pred_over_white=meta["sig_DM_pred"] / meta["sig_DM_pred_white"],
                   sig2_peak=float(np.mean(a["sig2_peak_ratio"])),
                   sig2_high=float(np.mean(a["sig2_high_ratio"])))
        line = (f"{tag:6s} S/N {snr:3d}: predicted sigma_DM x{row['pred_over_white']:.2f} of white; "
                f"sigma_t^2 est/true {row['sig2_peak']:.2f} at peak, {row['sig2_high']:.2f} where high |")
        for model in ["est", "true"]:
            d_fit, sig_DM = a[f"d_fit_{model}"], a[f"sig_DM_{model}"]
            ok = np.isfinite(d_fit) & np.isfinite(sig_DM)
            n = int(ok.sum())
            sig_mc = float(np.std(d_fit[ok]))
            row[model] = dict(
                n=n, n_failed=int((~ok).sum()), sig_mc=sig_mc,
                q=float(np.median(sig_DM[ok]) / sig_mc), q_err=float(1 / np.sqrt(2 * (n - 1))),
                sp=float(np.median(a[f"sig_S_{model}"] / a["Sp_max"]) / Sp_scatter),
                cover=float(np.mean(np.abs(d_fit[ok] - d0) <= sig_DM[ok])),
                pull=float(np.std((d_fit[ok] - d0) / sig_DM[ok])))
            m = row[model]
            line += (f" {model}: q {m['q']:.3f}+-{m['q_err']:.3f} sp {m['sp']:.3f} cover {m['cover']:.3f} "
                     f"pull {m['pull']:.3f} (failed {m['n_failed']}) |")
        rows.append(row)
        print(line)

    def span(f):
        v = [f(r) for r in rows]
        return [float(min(v)), float(max(v))]

    paper = dict(pred_over_white=span(lambda r: r["pred_over_white"]),
                 sig2_peak=span(lambda r: r["sig2_peak"]), sig2_high=span(lambda r: r["sig2_high"]))
    for model in ["est", "true"]:
        for k in ["q", "sp", "cover", "pull"]:
            paper[f"{k}_{model}"] = span(lambda r: r[model][k])
    json.dump(dict(rows=rows, paper=paper), open(sims.results_path("selfnoise_numbers.json"), "w"), indent=1)
    print(json.dumps(paper, indent=1))


if __name__ == "__main__":
    main()
