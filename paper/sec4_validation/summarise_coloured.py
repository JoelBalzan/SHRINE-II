"""The coloured-noise Monte Carlo (run_mc_coloured.py): sigma_S and sigma_DM against the scatter.

For each spectrum, S/N and way of evaluating the noise (white: the paper's estimate of sigma_t^2
from the profile; diag: the true variance per sample, correlations ignored; full: the true
Cov(n)):
    q        median quoted sigma_DM / the standard deviation of that model's d_fit, with the
             sampling error 1 / sqrt(2 (n - 1));
    sp       median sigma_S / Sp_max over the fractional scatter of Sp at the trial DM nearest d_0;
    cover    fraction of realisations with |d_fit - d_0| <= sigma_DM;
    pull     standard deviation of (d_fit - d_0) / sigma_DM, each realisation's d_fit against its
             own quoted sigma_DM (q can be below 1 while pull is 1 when sigma_DM changes between
             realisations and is larger where d_fit lands farther from d_0);
    p16, p84 16th and 84th percentiles of the quoted sigma_DM / its median (as in summarise.py);
    n_failed realisations whose window or fit failed;
for the full model also
    q_par    median sigma_DM / the scatter of d_par, the vertex without the second-order term;
    second   the scatter of d_fit / that of d_par: what the second-order term adds;
and sqrt(<||e||^2>)/||a|| at d_0 for the coloured noise and for white noise of the same S/N.

Writes results/coloured_numbers.json.

usage: python summarise_coloured.py   (after run_coloured.sh)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402

MODELS = ["white", "diag", "full"]


def main():
    rows = []
    for spectrum in ["ar1", "red"]:
        for snr in [15, 40, 200]:
            path = sims.results_path(f"coloured_{spectrum}_snr{snr}.npz")
            if not os.path.exists(path):
                continue
            a = np.load(path)
            meta = json.loads(str(a["meta"]))
            d0 = meta["d0"]
            Sp_scatter = np.std(a["Sp_d0"]) / np.mean(a["Sp_d0"])
            row = dict(spectrum=spectrum, snr=snr, kc=meta["kc"], e_over_a=meta["e_over_a"],
                       e_over_a_white=meta["e_over_a_white"], sig_DM_pred=meta["sig_DM_pred"])
            line = (f"{spectrum:3s} S/N {snr:3d}: sqrt(<||e||^2>)/||a|| {meta['e_over_a']:.2f} "
                    f"(white {meta['e_over_a_white']:.2f}) |")
            for model in MODELS:
                d_fit, sig_DM = a[f"d_fit_{model}"], a[f"sig_DM_{model}"]
                ok = np.isfinite(d_fit) & np.isfinite(sig_DM)
                n = int(ok.sum())
                sig_mc = float(np.std(d_fit[ok]))
                q = float(np.median(sig_DM[ok]) / sig_mc)
                sp = float(np.median(a[f"sig_S_{model}"] / a["Sp_max"]) / Sp_scatter)
                cover = float(np.mean(np.abs(d_fit[ok] - d0) <= sig_DM[ok]))
                pull = float(np.std((d_fit[ok] - d0) / sig_DM[ok]))
                p16, p84 = np.percentile(sig_DM[ok], [16, 84]) / np.median(sig_DM[ok])
                row[model] = dict(n=n, n_failed=int((~ok).sum()), sig_mc=sig_mc, q=q,
                                  q_err=float(1 / np.sqrt(2 * (n - 1))), sp=sp, cover=cover,
                                  pull=pull, p16=float(p16), p84=float(p84))
                line += (f" {model}: q {q:.2f} pull {pull:.2f} p16/84 {p16:.2f}/{p84:.2f} sp {sp:.2f} "
                         f"cover {cover:.2f} (failed {row[model]['n_failed']}) |")
            ok = np.isfinite(a["d_fit_full"]) & np.isfinite(a["sig_DM_full"]) & np.isfinite(a["d_par_full"])
            sig_par = np.std(a["d_par_full"][ok])
            row["full"]["q_par"] = float(np.median(a["sig_DM_full"][ok]) / sig_par)
            row["full"]["second"] = float(np.std(a["d_fit_full"][ok]) / sig_par)
            line += f" full without second order: q {row['full']['q_par']:.2f}, second-order factor {row['full']['second']:.2f}"
            rows.append(row)
            print(line)
    json.dump(rows, open(sims.results_path("coloured_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
