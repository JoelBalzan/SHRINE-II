"""Ordinary versus generalised least squares for the parabola fit (Section 3.3.4).

V (the covariance of Sp between trial DMs in the fit window) is close to singular: neighbouring
trial DMs share almost all their noise. GLS, c = (A^T V^-1 A)^-1 A^T V^-1 y, therefore depends on
the smallest eigenvalues of V, which the first-order noise model determines least well. This
script compares, on the noise realisations of run_mc.py for the one-component burst at S/N = 40:
    - the eigenvalue spectrum of V for the example burst;
    - the vertex scatter of OLS (the paper's d_fit) and of GLS with V^-1 replaced by a
      pseudo-inverse keeping eigenvalues above rcond times the largest, for several rcond;
    - each estimator's median predicted sigma_DM.
Writes results/gls_check.json.

usage: python gls_check.py [--n_real 200]
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402

RCONDS = [1e-2, 1e-3, 1e-4, 1e-6, 1e-9]


def g_neg(F):
    return np.any(np.diag(F) <= 0)


def gls_vertex(x, y, V, rcond):
    """Vertex and its predicted std for the GLS parabola fit, with V^-1 a pseudo-inverse."""
    A = np.column_stack([np.ones_like(x), x, x**2])
    Vi = np.linalg.pinv(V, rcond=rcond, hermitian=True)
    G = A.T @ Vi @ A
    # Fewer than three eigenvectors kept, or too few for the parabola: GLS is undefined.
    if np.linalg.matrix_rank(G, tol=1e-10 * np.abs(G).max()) < 3:
        return np.nan, np.nan
    F = np.linalg.inv(G)
    c = F @ A.T @ Vi @ y
    if c[2] >= 0 or g_neg(F):
        return np.nan, np.nan
    xv = -c[1] / (2 * c[2])
    # Predicted std of the vertex, to first order: about the vertex, delta x = -delta c1'/(2 c2),
    # with c1' = c1 + 2 c2 xv; var(c1') from F.
    g = np.array([0, 1, 2 * xv])
    return xv, np.sqrt(g @ F @ g) / abs(2 * c[2])


def main():
    parser = ArgumentParser()
    parser.add_argument("--n_real", type=int, default=200)
    args = parser.parse_args()

    ex = sims.example()
    res = dm_uncertainty(ex["P"], ex["d"], ex["pilot"]["kc"], ex["dt_nu"])
    w = np.linalg.eigvalsh(res["V"])[::-1]
    eig = (w[:8] / w[0]).tolist()

    I0, freq, time_ms = sims.load_burst("single")
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, 40)
    _, seed = sims.CASES[("single", 40)]
    rng = np.random.default_rng(seed)
    p = sims.pilot(I0, dt_nu, sc, rng)
    d = p["start"] + p["step"] * np.arange(p["n"])
    P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)

    ols, ols_s = [], []
    gls = {r: [] for r in RCONDS}
    gls_s = {r: [] for r in RCONDS}
    for _ in range(args.n_real):
        P = P0 + sims.dedisperse_grid(rng.normal(0, sc, I0.shape), p["start"], p["step"], p["n"],
                                      dt_nu)
        res = dm_uncertainty(P, d, p["kc"], dt_nu)
        if not res["ok"]:
            continue
        ols.append(res["d_fit"])
        ols_s.append(res["sig_DM"])
        x = d[res["win"]] - res["d_fit"]
        for rc in RCONDS:
            xv, s = gls_vertex(x, res["Sp"][res["win"]], res["V"], rc)
            gls[rc].append(res["d_fit"] + xv)
            gls_s[rc].append(s)

    out = dict(eig_over_largest=eig, n_win=int(res["n_win"]), n_real=len(ols),
               ols=dict(scatter=float(np.std(np.array(ols) - p["d0"])),
                        median_sig=float(np.median(ols_s))))
    for rc in RCONDS:
        v = np.array(gls[rc])
        ok = np.isfinite(v)
        out[f"gls_rcond{rc:g}"] = dict(scatter=float(np.std(v[ok] - p["d0"])),
                                       median_sig=float(np.nanmedian(gls_s[rc])),
                                       n_fail=int((~ok).sum()))
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "gls_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
