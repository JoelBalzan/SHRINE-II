"""Validity at low S/N (Section 5.2): how small the filtered noise is compared with the structure
vector in the simulations of Section 4, for both bursts at every S/N, at d_0 and with the k_c of
each case's pilot.

With s the noiseless profile at d_0, c_k its DCT coefficients (orthonormal DCT-II, k = 1 ... N) and
white profile noise of variance sigma^2 = n_chan sc^2: D1^T D1 is diagonal in the DCT basis, with
eigenvalues lam_k = 2 - 2 cos[(k - 1) pi / N], so mode k contributes
    lam_k f(k)^2 c_k^2      to ||a||^2 = ||D1 S s||^2, and
    lam_k f(k)^2 sigma^2    to <||e||^2> = <||D1 S n||^2>.
The numbers:
    e_over_a = sqrt(<||e||^2>) / ||a||                                           (Eq. small)
    noise_in_burst_modes = the fraction of <||e||^2> from modes with c_k^2 > sigma^2, where the
        burst's power exceeds the noise's;
    a_in_burst_modes = the same fraction of ||a||^2.

Writes results/lowsn_numbers.json.

usage: python lowsn_numbers.py
"""
import json
import os
import sys

import numpy as np
from scipy.fft import dct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from shrine2.structure import filter_response  # noqa: E402


def main():
    rows = []
    for tag in ["single", "double"]:
        I0, freq, time = sims.load_burst(tag)
        dt_nu = sims.channel_delays(freq, time[1] - time[0])
        N = time.size
        k = np.arange(1, N + 1)
        lam = 2 - 2 * np.cos((k - 1) * np.pi / N)
        for snr in sims.SNRS:
            _, seed = sims.CASES[(tag, snr)]
            sc = sims.channel_noise_sigma(I0, snr)
            p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed))
            f = filter_response(N, p["kc"])
            sigma2 = freq.size * sc**2

            c = dct(sims.dedisperse(I0, [p["d0"]], dt_nu)[0], norm="ortho")
            a2_per_mode = lam * f**2 * c**2
            e2_per_mode = lam * f**2 * sigma2
            burst_modes = c**2 > sigma2

            row = dict(tag=tag, snr=snr, kc=p["kc"],
                       e_over_a=float(np.sqrt(e2_per_mode.sum() / a2_per_mode.sum())),
                       noise_in_burst_modes=float(e2_per_mode[burst_modes].sum() / e2_per_mode.sum()),
                       a_in_burst_modes=float(a2_per_mode[burst_modes].sum() / a2_per_mode.sum()))
            rows.append(row)
            print(f"{tag:6s} S/N {snr:3d} k_c {p['kc']:3d}: sqrt<||e||^2>/||a|| {row['e_over_a']:.3f}, "
                  f"noise power in burst-dominated modes {row['noise_in_burst_modes']:.3f}, "
                  f"||a||^2 in them {row['a_in_burst_modes']:.4f}", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "lowsn_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
