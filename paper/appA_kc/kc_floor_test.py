"""k_c with a noise floor that depends on k (test for Appendix A; not used in the paper's runs).

S23 choose k_c as the first mode at which the six-mode running mean of M(k) = max over trial DMs of
|DCT_k| falls to M_n, the mean of M(k) over the upper half of the modes. For noise alone, M(k)
rises with k, so M_n lies above the noise level of M at the modes where k_c is found. This script
compares, on the noise realisations of run_mc_kc.py (same seeds and pilot):

    s23     S23's k_c (floor M_n);
    ext     the extended search of Appendix A;
    sim     the first k (>= 5) at which the running mean of M(k) falls to F(k), the mean of M(k)
            over N_FLOOR noise-only realisations with the same channel noise, dedispersed over the
            same coarse trial DMs (an observer would use the measured channel noise levels);
    off     the same with F(k) from one independent noise-only stretch of N samples per
            realisation (as from off-pulse data), smoothed over OFF_SMOOTH modes;
    sim_zZ  the first k at which the running mean of M(k) falls to that of noise alone plus Z times
            its standard deviation over the noise-only realisations (Z = 1, 2, 3): the first mode
            at which M(k) is consistent with noise, rather than at or below its mean.

For each case and rule: the distribution of k_c, the scatter of d_0(k_c) (d_0 of the noiseless burst
at each k_c), and the noiseless predicted sigma_DM at each k_c. Also M_n / F(k) at the median S23 k_c,
which measures how far the upper-half floor lies above the k-dependent one.

Writes results/kc_floor_<tag>_snr<snr>.json.

usage: python kc_floor_test.py TAG SNR [--n_real 200]
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
from scipy.fft import dct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sims  # noqa: E402

N_MEAN = 6  # S23's running mean
N_FLOOR = 20
OFF_SMOOTH = 21


def running_mean(x, n):
    """Trailing running mean over n modes, as in shrine2.structure.kc_search."""
    return np.convolve(x, np.ones(n) / n)[: x.size]


def centred_mean(x, n):
    """Centred running mean over n modes, with the ends averaged over fewer modes."""
    k = np.ones(n)
    return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")


def max_dct(I):
    return np.max(np.abs(dct(I, norm="ortho", axis=-1)), axis=0)


def kc_floor(M, F, Fbar=None):
    """First k >= N_MEAN - 1 at which the running mean of M falls to that of the floor F (or to
    Fbar, if given, which is already a running mean)."""
    Mbar = running_mean(M, N_MEAN)
    if Fbar is None:
        Fbar = running_mean(F, N_MEAN)
    below = np.nonzero(Mbar[N_MEAN - 1:] <= Fbar[N_MEAN - 1:])[0]
    return int(below[0] + N_MEAN - 1) if below.size else None


def main():
    parser = ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("snr", type=int)
    parser.add_argument("--n_real", type=int, default=200)
    args = parser.parse_args()
    tag, snr = args.tag, args.snr

    I0, freq, time_ms = sims.load_burst(tag)
    n_chan, N = I0.shape
    dt_nu = sims.channel_delays(freq, time_ms[1] - time_ms[0])
    sc = sims.channel_noise_sigma(I0, snr)
    _, seed = sims.CASES[(tag, snr)]
    rng = np.random.default_rng(seed)
    p = sims.pilot(I0, dt_nu, sc, rng)
    d = p["start"] + p["step"] * np.arange(p["n"])
    P0 = sims.dedisperse_grid(I0, p["start"], p["step"], p["n"], dt_nu)
    coarse = sims.coarse_grid(dt_nu)
    kcfile = np.load(sims.results_path(f"kc_{tag}_snr{snr}.npz"))

    # The k-dependent floor from noise alone.
    rng_f = np.random.default_rng(seed + 5000)
    Mnoise = np.array([max_dct(sims.dedisperse_grid(rng_f.normal(0, sc, I0.shape), *coarse, dt_nu))
                       for _ in range(N_FLOOR)])
    F = Mnoise.mean(axis=0)
    Mn_noise = Mnoise[:, N // 2:].mean()
    Fbars = np.array([running_mean(m, N_MEAN) for m in Mnoise])
    Fbar_mean, Fbar_std = Fbars.mean(axis=0), Fbars.std(axis=0, ddof=1)
    ZS = [1, 2, 3]

    kcs = {r: np.full(args.n_real, -1) for r in ["s23", "ext", "sim", "off"] + [f"sim_z{z}" for z in ZS]}
    for r in range(args.n_real):
        noise = rng.normal(0, sc, I0.shape)
        M = max_dct(sims.dedisperse_grid(I0 + noise, *coarse, dt_nu))
        Mbar = running_mean(M, N_MEAN)
        Mn = M[N // 2:].mean()
        k23 = next((i for i in range(N_MEAN - 1, N) if Mbar[i] <= Mn), -1)
        kcs["s23"][r] = k23
        kcs["ext"][r] = kcfile["kc_ext"][r] if np.isfinite(kcfile["kc_ext"][r]) else -1
        assert k23 == kcfile["kc_var"][r], (tag, snr, r, k23, kcfile["kc_var"][r])
        k = kc_floor(M, F)
        kcs["sim"][r] = -1 if k is None else k
        Moff = max_dct(sims.dedisperse_grid(rng_f.normal(0, sc, I0.shape), *coarse, dt_nu))
        k = kc_floor(M, centred_mean(Moff, OFF_SMOOTH))
        kcs["off"][r] = -1 if k is None else k
        for z in ZS:
            k = kc_floor(M, None, Fbar_mean + z * Fbar_std)
            kcs[f"sim_z{z}"][r] = -1 if k is None else k

    cache = {}

    def pred(k):
        if k not in cache:
            nf = sims.noiseless_fit(P0, d, int(k), dt_nu, p["sig2"])
            cache[k] = (nf["d0"], nf["sig_DM"])
        return cache[k]

    out = dict(tag=tag, snr=snr, n_real=args.n_real, kc_pilot=p["kc"],
               floor_ratio_at_kc=float(Mn_noise / F[int(np.median(kcs["s23"]))]),
               floor_ratio_k10_30=float(Mn_noise / F[10:31].mean()), rules={})
    for rule, ks in kcs.items():
        ok = ks > 0
        d0s = np.array([pred(k)[0] for k in ks[ok]])
        sig = np.array([pred(k)[1] for k in ks[ok]])
        out["rules"][rule] = dict(
            n_failed=int((~ok).sum()), kc_median=float(np.median(ks[ok])),
            kc_p16=float(np.percentile(ks[ok], 16)), kc_p84=float(np.percentile(ks[ok], 84)),
            kc_min=int(ks[ok].min()), kc_max=int(ks[ok].max()),
            std_d0=float(np.std(d0s)), median_sig_DM=float(np.median(sig)),
            std_d0_over_sig=float(np.std(d0s) / np.median(sig)),
            frac_differs_from_s23=float(np.mean(ks != kcs["s23"])))
    path = os.path.join(HERE, "results", f"kc_floor_{tag}_snr{snr}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(tag, snr, f"M_n/F(k_c) {out['floor_ratio_at_kc']:.2f}", flush=True)
    for rule, v in out["rules"].items():
        print(f"  {rule:7s} k_c {v['kc_median']:.0f} [{v['kc_p16']:.0f}-{v['kc_p84']:.0f}] "
              f"({v['kc_min']}-{v['kc_max']}), std d0 / sigma_DM {v['std_d0_over_sig']:.2f}, "
              f"median pred sigma_DM {v['median_sig_DM']:.5f}, failed {v['n_failed']}", flush=True)


if __name__ == "__main__":
    main()
