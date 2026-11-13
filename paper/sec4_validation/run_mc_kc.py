"""Monte Carlo with k_c chosen from each realisation (end of Section 4 and Appendix A).

The same bursts, pilot, seeds and noise realisations as run_mc.py, whose results (k_c fixed) must
exist. For each realisation, k_c is chosen from that realisation's coarse dedispersed profiles:
    var: by S23's procedure (shrine2.choose_kc with extend=False);
    ext: with the extended search of Appendix A (extend=True);
and d_fit and sigma_DM are found again with that k_c (shrine2.dm_uncertainty on the same profiles).
Also records d_0(k_c), the noiseless burst's structure-maximising DM for each k_c (sims.noiseless_fit),
since d_0 moves with k_c.

Writes results/kc_<tag>_snr<snr>.npz.

usage: python run_mc_kc.py tag snr
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sims  # noqa: E402
from shrine2 import choose_kc, dm_uncertainty  # noqa: E402


def main():
    args = get_args()
    tag, snr = args.tag, args.snr
    fixed = np.load(sims.results_path(f"mc_{tag}_snr{snr}.npz"))
    meta = json.loads(str(fixed["meta"]))
    n_real, seed = meta["n_real"], meta["seed"]

    I0, freq, time_ms = sims.load_burst(tag)
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)
    rng = np.random.default_rng(seed)
    p = sims.pilot(I0, dt_nu, sc, rng)
    kc = p["kc"]
    assert kc == meta["kc"] and np.isclose(p["d0"], meta["d0"]), "pilot differs from run_mc.py"

    d = p["start"] + p["step"] * np.arange(p["n"])
    P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)
    coarse = sims.coarse_grid(dt_nu)

    # d_0 for each k_c met, from the noiseless profiles (cached).
    d0_of_kc = {}

    def d0_for(k):
        if k not in d0_of_kc:
            d0_of_kc[k] = sims.noiseless_fit(P0, d, k, dt_nu, p["sig2"])["d0"]
        return d0_of_kc[k]

    def fit(P, k):
        """d_fit and sigma_DM with cutoff k (nan if the fit fails)."""
        res = dm_uncertainty(P, d, k, dt_nu)
        if res["ok"]:
            return res["d_fit"], res["sig_DM"]
        return np.nan, np.nan

    keys = ["kc_var", "d_var", "s_var", "d0_var", "kc_ext", "d_ext", "s_ext", "d0_ext"]
    out = {}
    for k in keys:
        out[k] = np.full(n_real, np.nan)

    t0 = time.time()
    for r in range(n_real):
        noise = rng.normal(0, sc, I0.shape)
        P = P0 + sims.dedisperse_grid(noise, p["start"], p["step"], p["n"], dt_nu)
        fixed_fit = (fixed["d_fit"][r], fixed["sig_DM"][r])

        # k_c from this realisation's coarse profiles, by S23's procedure and with the extension.
        I_coarse = sims.dedisperse_grid(I0 + noise, *coarse, dt_nu)
        k_var = choose_kc(I_coarse, extend=False)
        try:
            k_ext = choose_kc(I_coarse, extend=True)
        except RuntimeError:
            k_ext = None

        # Refit only where k_c changed.
        if k_var == kc:
            var_fit = fixed_fit
        else:
            var_fit = fit(P, k_var)
        out["kc_var"][r] = k_var
        out["d_var"][r], out["s_var"][r] = var_fit
        out["d0_var"][r] = d0_for(k_var)

        if k_ext is not None:
            if k_ext == k_var:
                ext_fit = var_fit
            elif k_ext == kc:
                ext_fit = fixed_fit
            else:
                ext_fit = fit(P, k_ext)
            out["kc_ext"][r] = k_ext
            out["d_ext"][r], out["s_ext"][r] = ext_fit
            out["d0_ext"][r] = d0_for(k_ext)

        if (r + 1) % 100 == 0:
            print(f"  {r + 1}/{n_real}, {time.time() - t0:.0f} s", flush=True)

    d0_table = {}
    for k in sorted(d0_of_kc):
        d0_table[str(k)] = d0_of_kc[k]
    meta_kc = dict(meta, runtime_kc=time.time() - t0, d0_of_kc=d0_table)
    path = sims.results_path(f"kc_{tag}_snr{snr}.npz")
    np.savez_compressed(path, meta=json.dumps(meta_kc), d_fix=fixed["d_fit"],
                        s_fix=fixed["sig_DM"], **out)
    print(f"wrote {path}", flush=True)


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag", choices=["single", "double"])
    parser.add_argument("snr", type=int, choices=sims.SNRS)
    return parser.parse_args()


if __name__ == "__main__":
    main()
