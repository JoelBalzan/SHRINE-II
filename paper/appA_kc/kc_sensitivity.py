"""Sensitivity of the extended k_c search (Appendix A) to its three settings.

The extended search uses a six-mode running mean, a rise level of M_n + 4 sig_Mbar and a search
from k_c to 2 k_c. For the noise realisations of run_mc_kc.py (same seeds, same pilot), this
script repeats the search with the level at 3, 4 and 5 sig_Mbar, the running mean over 4, 6 and 8
modes, and the search to 1.5 k_c, 2 k_c and 3 k_c (one setting changed at a time), and records
k_c. Where k_c differs from the default search, d_fit and sigma_DM are found again with that k_c,
and the scatter of d_fit about d_0(k_c) is compared with the median sigma_DM, as in Section 4.

Run for the cases in which the extended search changes k_c at all (the two-component burst at
S/N = 25 and 40) and, as a control, the one-component burst at S/N = 40.

Writes results/kc_sensitivity.json.

usage: python kc_sensitivity.py
"""
import json
import os
import sys

import numpy as np
from scipy.fft import dct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402
from shrine2 import dm_uncertainty  # noqa: E402

CASES = [("double", 40), ("double", 25), ("single", 40)]
DEFAULT = dict(n_sig=4, n_mean=6, reach=2.0)
VARIANTS = {"default": DEFAULT,
            "n_sig=3": dict(DEFAULT, n_sig=3), "n_sig=5": dict(DEFAULT, n_sig=5),
            "n_mean=4": dict(DEFAULT, n_mean=4), "n_mean=8": dict(DEFAULT, n_mean=8),
            "reach=1.5": dict(DEFAULT, reach=1.5), "reach=3": dict(DEFAULT, reach=3.0)}


def kc_ext(I, n_sig, n_mean, reach):
    """shrine2.structure.kc_search with its settings exposed (S23's k_c uses the same n_mean)."""
    M = np.max(np.abs(dct(I, norm="ortho", axis=-1)), axis=0)
    N = M.size
    Mbar = np.convolve(M, np.ones(n_mean) / n_mean)[:N]
    Mn = np.mean(M[N // 2:])
    level = Mn + n_sig * np.std(Mbar[N // 2 + n_mean:])
    kc = next((i for i in range(n_mean - 1, N) if Mbar[i] <= Mn), None)
    if kc is None:
        return None
    rises = [i for i in range(kc, min(int(reach * kc), N // 2)) if Mbar[i] > level]
    if not rises:
        return kc
    return next((i for i in range(rises[-1], N) if Mbar[i] <= Mn), None)


def main():
    out = {}
    for tag, snr in CASES:
        n_real, seed = sims.CASES[(tag, snr)]
        I0, freq, time_ms = sims.load_burst(tag)
        dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
        sc = sims.channel_noise_sigma(I0, snr)
        rng = np.random.default_rng(seed)
        p = sims.pilot(I0, dt_nu, sc, rng)
        d = p["start"] + p["step"] * np.arange(p["n"])
        P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)
        coarse = sims.coarse_grid(dt_nu)
        kcfile = np.load(sims.results_path(f"kc_{tag}_snr{snr}.npz"))

        d0_cache, fit_cache = {}, {}

        def d0_for(k):
            if k not in d0_cache:
                d0_cache[k] = sims.noiseless_fit(P0, d, k, dt_nu, p["sig2"])["d0"]
            return d0_cache[k]

        kcs = {v: np.full(n_real, -1) for v in VARIANTS}
        dfit = {v: np.full(n_real, np.nan) for v in VARIANTS}
        sdm = {v: np.full(n_real, np.nan) for v in VARIANTS}
        d0v = {v: np.full(n_real, np.nan) for v in VARIANTS}
        for r in range(n_real):
            noise = rng.normal(0, sc, I0.shape)
            Ic = sims.dedisperse_grid(I0 + noise, *coarse, dt_nu)
            ks = {v: kc_ext(Ic, **s) for v, s in VARIANTS.items()}
            # The default must reproduce run_mc_kc.py's extended k_c.
            ref = kcfile["kc_ext"][r]
            assert (ks["default"] is None and np.isnan(ref)) or ks["default"] == ref, (tag, snr, r)
            need = {k for k in ks.values() if k is not None}
            P = None
            for k in need:
                if k == ks["default"] and np.isfinite(kcfile["d_ext"][r]):
                    fit_cache[(r, k)] = (kcfile["d_ext"][r], kcfile["s_ext"][r])
                    continue
                if P is None:
                    P = P0 + sims.dedisperse_grid(noise, p["start"], p["step"], p["n"], dt_nu)
                res = dm_uncertainty(P, d, k, dt_nu)
                fit_cache[(r, k)] = (res["d_fit"], res["sig_DM"]) if res["ok"] else (np.nan, np.nan)
            for v, k in ks.items():
                if k is None:
                    continue
                kcs[v][r] = k
                dfit[v][r], sdm[v][r] = fit_cache[(r, k)]
                d0v[v][r] = d0_for(k)
            fit_cache = {}

        case = {}
        for v in VARIANTS:
            ok = np.isfinite(dfit[v])
            scatter = np.std(dfit[v][ok] - d0v[v][ok])
            case[v] = dict(frac_differs=float(np.mean(kcs[v] != kcs["default"])),
                           kc_median=float(np.median(kcs[v][kcs[v] >= 0])),
                           scatter_about_d0kc_over_median_sig=float(scatter / np.median(sdm[v][ok])),
                           scatter_about_fixed_d0_over_fixed_scatter=float(
                               np.std(dfit[v][ok] - p["d0"]) /
                               np.nanstd(kcfile["d_fix"] - p["d0"])))
        out[f"{tag}_snr{snr}"] = case
        print(tag, snr, json.dumps({v: [round(c["frac_differs"], 3),
                                        round(c["scatter_about_d0kc_over_median_sig"], 3),
                                        round(c["scatter_about_fixed_d0_over_fixed_scatter"], 3)]
                                    for v, c in case.items()}), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "kc_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
