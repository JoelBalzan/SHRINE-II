"""Why S23's uncertainty is too large (Section 4, with Section 2.4): the numbers, for both bursts
at every S/N, at d_0 and with the k_c of each case's pilot.

With s the noiseless profile at d_0 and white profile noise of variance sigma^2 = n_chan sc^2:
    a = D1 S s and sigma_S = ||w|| sigma (Eq. sigS), with w from s (shrine2.structure.coefficients);
    <||e||^2> = sigma^2 ||D1 S||_F^2, the filtered noise, e = D1 S n;
    d_s = D1 S (s - S s), the burst part of S23's residual D1 S Delta_x;
    <||d_n||^2> = sigma^2 ||D1 S (I - S)||_F^2, its noise part (Eq. dsdn);
and from these:
    frac_along_a_hat = sigma_S^2 / <||e||^2>, the fraction of the filtered noise power along a_hat;
    e_over_sigma_S = sqrt(<||e||^2>) / sigma_S;
    frac_d_s = ||d_s||^2 / (||d_s||^2 + <||d_n||^2>), the burst's share of <||D1 S Delta_x||^2>;
    d_n_over_sigma_S = sqrt(<||d_n||^2>) / sigma_S.
||M||_F^2 (the sum of M's squared elements) is the trace of M M^T, so <||M n||^2> = sigma^2 ||M||_F^2
for white noise.

Writes results/s23_residual_numbers.json.

usage: python s23_residual_numbers.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from shrine2.noise import sigma_S  # noqa: E402
from shrine2.structure import coefficients, forward_difference_matrix, smooth  # noqa: E402


def main():
    rows = []
    for tag in ["single", "double"]:
        I0, freq, time = sims.load_burst(tag)
        dt_nu = sims.channel_delays(freq, time[1] - time[0])
        N = time.size
        D1 = forward_difference_matrix(N)
        identity = np.eye(N)
        for snr in sims.SNRS:
            _, seed = sims.CASES[(tag, snr)]
            sc = sims.channel_noise_sigma(I0, snr)
            p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed))
            kc = p["kc"]
            sigma2 = freq.size * sc**2

            s = sims.dedisperse(I0, [p["d0"]], dt_nu)[0]
            Ss = smooth(s, kc)
            norm_a = np.linalg.norm(D1 @ Ss)
            sig_S = sigma_S(coefficients(s, kc), np.full(N, sigma2))

            # S as a matrix: smooth() applied to each row of the identity gives the rows of S
            # (S is symmetric). Then D1 S and D1 S (I - S).
            S = smooth(identity, kc)
            D1S = D1 @ S
            D1S_residual = D1S @ (identity - S)
            e2 = sigma2 * np.sum(D1S**2)
            d_s2 = np.linalg.norm(D1 @ smooth(s - Ss, kc)) ** 2
            d_n2 = sigma2 * np.sum(D1S_residual**2)

            row = dict(tag=tag, snr=snr, kc=kc, d0=p["d0"], norm_a=float(norm_a), sig_S=float(sig_S),
                       frac_along_a_hat=float(sig_S**2 / e2), e_over_sigma_S=float(np.sqrt(e2) / sig_S),
                       frac_d_s=float(d_s2 / (d_s2 + d_n2)), d_s_over_sigma_S=float(np.sqrt(d_s2) / sig_S),
                       d_n_over_sigma_S=float(np.sqrt(d_n2) / sig_S),
                       s23_over_sigma_S=float(np.sqrt(d_s2 + d_n2) / sig_S))
            rows.append(row)
            print(f"{tag:6s} S/N {snr:3d} k_c {kc:3d}: noise along a_hat {row['frac_along_a_hat']:.3f}, "
                  f"||e||/sigma_S {row['e_over_sigma_S']:.2f}, d_s share {row['frac_d_s']:.2f}, "
                  f"d_n/sigma_S {row['d_n_over_sigma_S']:.2f}", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "s23_residual_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
