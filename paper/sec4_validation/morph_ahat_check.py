"""Why sigma_DM exceeds the scatter for the drifting burst at S/N = 200 (Section 4).

For the first N_REAL realisations of run_mc_review.py morph drift020 200 (same seed and pilot),
compares the scatter of d_fit about d_0 with three versions of the median quoted sigma_DM:
    data          shrine2.dm_uncertainty with the true sigma_t^2 (as quoted, but sigma_t^2 exact);
    noiseless w   the same B and c2, but V from w(d_j) of the noiseless profiles;
    noiseless c2  the same V, with the noiseless burst's c2.
Also the noiseless prediction (sims.noiseless_fit) over the scatter. Writes
results/morph_ahat_check.json.

usage: python morph_ahat_check.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sims  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402
from shrine2.covariance import noise_covariance  # noqa: E402
from shrine2.structure import coefficients  # noqa: E402

N_REAL = 80
TAG, SNR, SEED = "drift020", 200, 700 + 200  # run_mc_review.py's seed for this case


def main():
    I0, freq, tm = sims.load_burst(TAG)
    dt_nu = sims.channel_delays(freq, tm[1] - tm[0])
    sc = sims.channel_noise_sigma(I0, SNR)
    rng = np.random.default_rng(SEED)
    p = sims.pilot(I0, dt_nu, sc, rng)
    kc = p["kc"]
    d = p["start"] + p["step"] * np.arange(p["n"])
    P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)
    nf = sims.noiseless_fit(P0, d, kc, dt_nu, p["sig2"])

    q_data, q_w0, q_c20, d_fit = [], [], [], []
    for _ in range(N_REAL):
        P = P0 + sims.dedisperse_grid(rng.normal(0, sc, I0.shape), p["start"], p["step"], p["n"],
                                      dt_nu)
        res = dm_uncertainty(P, d, kc, dt_nu, cov_n=p["sig2"])
        if not res["ok"]:
            continue
        win, B, c = res["win"], res["B"], res["c"]
        V0 = noise_covariance(coefficients(P0[win], kc), p["sig2"],
                              np.outer(d[win] - d[res["j_max"]], dt_nu))
        q_data.append(res["sig_DM"])
        q_w0.append(np.sqrt(B[1] @ V0 @ B[1]) / abs(2 * c[2]))
        q_c20.append(np.sqrt(B[1] @ res["V"] @ B[1]) / abs(2 * nf["c2"]))
        d_fit.append(res["d_fit"])
    scatter = np.std(np.array(d_fit) - nf["d0"])
    out = dict(n=len(d_fit), kc=kc, scatter=float(scatter),
               q_data=float(np.median(q_data) / scatter),
               q_noiseless_w=float(np.median(q_w0) / scatter),
               q_noiseless_c2=float(np.median(q_c20) / scatter),
               q_noiseless_prediction=float(nf["sig_DM"] / scatter))
    print(json.dumps(out, indent=1))
    json.dump(out, open(sims.results_path("morph_ahat_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
