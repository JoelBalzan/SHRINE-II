"""Accuracy of the profile-noise variance from the high DCT modes (Section 3.3.2).

For N = 601 and k_min = N/4 (the simulations of Section 4): the diagonal of the projection L onto
the removed modes, the concentration of its rows, and the variance of the reconstruction
(Eq. varhigh) divided by 1 - k_min/N, for a variance that rises fivefold in a Gaussian of five
samples standard deviation; and, for the same variance, the expected 15-sample estimate of Section
3.3.2 at the peak over the true variance there, with and without the reconstruction. Writes results/dct_variance_numbers.json.
"""
import json
import os

import numpy as np
from scipy.fft import dct

HERE = os.path.dirname(os.path.abspath(__file__))
N = 601
K_MIN = N // 4


def main():
    C = dct(np.eye(N), norm="ortho", axis=0)  # rows are the DCT-II modes
    l = np.zeros(N)
    l[:K_MIN] = 1
    L = C.T @ np.diag(l) @ C
    Ltt = np.diag(L)
    inner = slice(10, N - 10)
    ratio = (1 - Ltt) / (1 - K_MIN / N)

    # Fraction of the squared elements of row t (centre) within 3 samples of t.
    t = N // 2
    frac3 = np.sum(L[t, t - 3:t + 4] ** 2) / np.sum(L[t] ** 2)

    # Variance rising fivefold in a Gaussian of 5 samples standard deviation, at the centre.
    tt = np.arange(N)
    s2 = 1 + 4 * np.exp(-0.5 * ((tt - N / 2) / 5) ** 2)
    I_L = np.eye(N) - L
    var = (I_L**2) @ s2 / (1 - K_MIN / N)
    box = np.ones(15) / 15
    c = N // 2
    est15 = np.convolve(var, box, "same")[c] / s2[c]
    avg_only = np.convolve(s2, box, "same")[c] / s2[c]
    out = dict(N=N, k_min=K_MIN,
               ratio_interior_max_dev=float(np.max(np.abs(ratio[inner] - 1))),
               ratio_first_sample_dev=float(ratio[0] - 1),
               row_fraction_within_3=float(frac3),
               bump5_max_rel_err=float(np.max(np.abs(var / s2 - 1)[inner])),
               bump5_est15_over_true_peak=float(est15),
               bump5_avg15_only_over_true_peak=float(avg_only))
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "dct_variance_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
