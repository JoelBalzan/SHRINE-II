"""The derivative of e_par at d_0 against the fitted slope c1 (Section 3.3.4): the numbers.

sigma_DM could be written with the derivative e_par'(d_0) in place of std[c1], the slope of the
parabola fitted over the window. The two agree only if e_par(d) is linear across the window. For the
one-component burst at S/N = 15, 40 and 200 (k_c and d_0 from each case's pilot), with the noise
known and w(t; d) from the noiseless burst:
    e_par(d) = w(d) n(d) at trial DMs across +-2 h about d_0, where h = sqrt(2 sigma_S / |2 c2|) is
        the half-width over which ||a(d)|| falls by sigma_S (the noiseless fit window), with c2 from
        a parabola fitted to ||a(d)|| over +-3 sigma_DM;
    the derivative at d_0, by a central difference over +-h/1000;
    c1, the slope of a parabola fitted to e_par over |d - d_0| <= h (Eq. lsq).
The paper quotes std[derivative] / std[c1] - 1.

Writes results/slope_numbers.json.

usage: python slope_numbers.py [--n_real 600]
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from shrine2.fit import parabola_least_squares  # noqa: E402
from shrine2.noise import sigma_S  # noqa: E402
from shrine2.structure import coefficients, structure_parameter  # noqa: E402

SEED = 1234


def main():
    args = get_args()
    I0, freq, time = sims.load_burst("single")
    dt_nu = sims.channel_delays(freq, time[1] - time[0])
    rows = []
    for snr in [15, 40, 200]:
        _, seed = sims.CASES[("single", snr)]
        sc = sims.channel_noise_sigma(I0, snr)
        p = sims.pilot(I0, dt_nu, sc, np.random.default_rng(seed))
        kc, d0 = p["kc"], p["d0"]
        sig2 = p["sig2"]

        # The curvature of ||a(d)|| at d_0, sigma_S there and the window half-width h.
        d_near = d0 + np.linspace(-3, 3, 41) * p["sig_DM"]
        _, c = parabola_least_squares(d_near - d0, structure_parameter(sims.dedisperse(I0, d_near, dt_nu), kc))
        curvature = 2 * c[2]
        sig_S_d0 = sigma_S(coefficients(sims.dedisperse(I0, [d0], dt_nu)[0], kc), sig2)
        h = np.sqrt(2 * sig_S_d0 / abs(curvature))

        # Trial DMs across +-2h, and the pair for the derivative; w(t; d) of the noiseless burst.
        x = np.linspace(-2, 2, 161) * h
        eps = 1e-3 * h
        W = coefficients(sims.dedisperse_grid(I0, d0 + x[0], x[1] - x[0], x.size, dt_nu), kc)
        W_pair = coefficients(sims.dedisperse(I0, [d0 - eps, d0 + eps], dt_nu), kc)
        in_window = np.abs(x) <= h
        B, _ = parabola_least_squares(x[in_window], np.zeros(in_window.sum()))

        rng = np.random.default_rng(SEED)
        derivative = np.empty(args.n_real)
        c1 = np.empty(args.n_real)
        for r in range(args.n_real):
            noise = rng.normal(0, sc, I0.shape)
            e_par = np.sum(W * sims.dedisperse_grid(noise, d0 + x[0], x[1] - x[0], x.size, dt_nu), axis=1)
            e_pair = np.sum(W_pair * sims.dedisperse(noise, [d0 - eps, d0 + eps], dt_nu), axis=1)
            derivative[r] = (e_pair[1] - e_pair[0]) / (2 * eps)
            c1[r] = B[1] @ e_par[in_window]

        row = dict(snr=snr, kc=kc, d0=d0, h=h, curvature=curvature, sig_S=sig_S_d0,
                   std_derivative=float(derivative.std()), std_c1=float(c1.std()),
                   ratio=float(derivative.std() / c1.std()), n_real=args.n_real)
        rows.append(row)
        print(f"S/N {snr:3d}: k_c {kc}, h {h:.4f}; std[derivative] {row['std_derivative']:.3f}, "
              f"std[c1] {row['std_c1']:.3f}, ratio {row['ratio']:.3f}", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "slope_numbers.json"), "w"), indent=1)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n_real", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    main()
