"""S/N maximisation, as SHRINE's maximise_sn.py. Not part of the paper's method; kept for comparison.

For each trial DM, the smoothed profile S x (Eq. smooth) less a common baseline is searched for the boxcar, of any
start and of length 1 sample to 10 ms, with the largest S/N = (sum over the boxcar) / (sigma sqrt(length)).
The S/N-maximising DM is the trial DM with the largest boxcar S/N, and its range is where the S/N stays within 1
of that maximum (SHRINE's criterion).

SHRINE's choices are kept:
    sigma    = standard deviation of the residual I - S I over all trial DMs and samples;
    baseline = the smallest value of S I over all trial DMs.
Fixed relative to SHRINE: the S/N of a boxcar of L samples is now its sum of L samples over sigma sqrt(L) (SHRINE summed
L + 1 samples and reported a length of L - 1), boxcars may end at the last sample, and the range stops at the
ends of the trial DMs instead of running past them.
"""
import numpy as np

from .structure import smooth

MAX_WINDOW_MS = 10          # longest boxcar, as SHRINE


def best_boxcars(X, sigma, n_max):
    """For each row of X, the boxcar with the largest S/N over all starts and lengths 1..n_max.

    Returns arrays (S/N, start, length), one entry per row.

    Args:
        X: baseline-subtracted smoothed profiles, shape (n_d, N).
        sigma: noise standard deviation per sample.
        n_max: longest boxcar, in samples.
    Returns:
        sn, start, length: each shape (n_d,).
    """
    n, N = X.shape
    C = np.concatenate([np.zeros((n, 1)), np.cumsum(X, axis=1)], axis=1)     # C[:, i] = sum of the first i samples
    sn_best, start, length = np.full(n, -np.inf), np.zeros(n, int), np.ones(n, int)
    rows = np.arange(n)
    for L in range(1, min(n_max, N) + 1):
        sn = (C[:, L:] - C[:, :-L]) / (sigma * np.sqrt(L))   # boxcars of L samples at every start
        i = np.argmax(sn, axis=1)
        better = sn[rows, i] > sn_best
        sn_best[better], start[better], length[better] = sn[rows, i][better], i[better], L
    return sn_best, start, length


def maximise_sn(I, d, kc, dt):
    """The S/N-maximising trial DM of I(d_j, t) and the trial DMs within 1 of its S/N. Returns a dict.

    Args:
        I: I(d_j, t), shape (n_d, N).
        d: the trial DMs, shape (n_d,), pc cm^-3.
        kc: filter cutoff k_c.
        dt: time resolution, us (sets the 10-ms longest boxcar).
    Returns:
        dict with
            d, SN, kc, sigma, Sx: the trial DMs, the best boxcar S/N at each, k_c, the noise sigma, S x;
            j_max, d_sn, SN_max: the trial DM with the largest S/N;
            start, length: its boxcar, in samples;
            lo, hi, lo_found, hi_found: indices of the first trial DMs where the S/N has dropped by 1, and whether it did.
    """
    Sx = smooth(I, kc)
    sigma = np.std(I - Sx)                               # SHRINE: residual over all trial DMs and samples
    X = Sx - Sx.min()                                    # SHRINE: common baseline
    SN, start, length = best_boxcars(X, sigma, int(MAX_WINDOW_MS * 1000 / dt))

    j = int(np.argmax(SN))
    lo = hi = j
    while lo > 0 and SN[j] - SN[lo] < 1:                 # walk out until the S/N has dropped by 1
        lo -= 1
    while hi < SN.size - 1 and SN[j] - SN[hi] < 1:
        hi += 1
    return dict(d=np.asarray(d), SN=SN, kc=kc, sigma=sigma, Sx=Sx, j_max=j, d_sn=d[j], SN_max=SN[j],
                start=int(start[j]), length=int(length[j]), lo=lo, hi=hi,
                lo_found=SN[j] - SN[lo] >= 1, hi_found=SN[j] - SN[hi] >= 1)
