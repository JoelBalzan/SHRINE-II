"""The profile-noise variance sigma_t^2 and the structure parameter uncertainty sigma_S.

Paper: Section 3.3.2 (sigma_t^2) and Section 3.1 (Eq. sigS).
"""
import numpy as np
from scipy.fft import dct, idct
from scipy.ndimage import uniform_filter1d

from .structure import filter_response


def noise_variance(x, kc, n_avg=15, verbose=False):
    """sigma_t^2 at every sample of one profile x (Section 3.3.2).

    1. Keep only the DCT modes k > k_min, where k_min = max(N/4, 2 k_c), at most 3N/4. These lie
       above the burst's power, so the reconstruction holds only noise.
    2. Average its square over n_avg = 15 samples centred on t (end samples repeated at the edges).
    3. Divide by 1 - k_min/N: white noise keeps only that fraction of its variance in those modes.

    Args:
        x: one profile, shape (N,) (in dm_uncertainty, the profile at d_max).
        kc: filter cutoff k_c, which sets k_min.
        n_avg: number of samples in the running mean square.
        verbose: print k_min and the range of sigma_t^2.
    Returns:
        sigma_t^2 at every sample, shape (N,), in the units of x squared.
    """
    N = x.size

    # k_min = max(N/4, 2 k_c), with 2 k_c capped at 3N/4 so that at least N/4 modes are kept.
    k_min = max(N // 4, min(2 * kc, 3 * N // 4))

    # 1. Remove modes k = 1..k_min (array indices 0..k_min-1): what is left is noise only.
    c = dct(x, norm="ortho")
    c[:k_min] = 0
    noise_only = idct(c, norm="ortho")

    # 2. The local mean square of the noise-only profile.
    mean_square = uniform_filter1d(noise_only**2, size=n_avg, mode="nearest")

    # 3. Correct for the variance removed with the low modes.
    sig2 = mean_square / (1 - k_min / N)

    if verbose:
        print(f"  sigma_t^2 (Sec. 3.3.2): from the DCT modes k > k_min = {k_min} of N = {N}; "
              f"mean {np.mean(sig2):.4g}, range {sig2.min():.4g} to {sig2.max():.4g}")
    return sig2


def sigma_S(w, sig2):
    """sigma_S = sqrt( sum_t w_t^2 sigma_t^2 )   (Eq. sigS).

    Args:
        w: coefficients at one trial DM, shape (N,) (from structure.coefficients).
        sig2: sigma_t^2, shape (N,) (from noise_variance).
    Returns:
        sigma_S, a float in the units of Sp.
    """
    return np.sqrt(np.sum(w**2 * sig2))


def noise_ratio(sig2, kc, Sp):
    """Typical length of the filtered noise relative to the structure parameter,
    sqrt(<||e||^2>) / Sp.

    The expansion behind the uncertainties needs ||e|| << ||a|| (Eq. small); Section 5.2 discusses
    where it holds (the simulations span ~0.3 at S/N 15 to ~0.07 at S/N 200). For noise
    uncorrelated between samples, <||e||^2> = trace(D1 S diag(sigma_t^2) S D1^T)
    ~ mean(sigma_t^2) sum_k lam_k f(k)^2 (exact for constant sigma_t^2). This is a diagnostic
    only: it is not used in sigma_S or sigma_DM.

    Args:
        sig2: sigma_t^2, shape (N,).
        kc: filter cutoff k_c.
        Sp: the structure parameter at d_max.
    Returns:
        sqrt(<||e||^2>) / Sp, a float.
    """
    f, lam = filter_response(sig2.size, kc)

    # <||e||^2> ~ mean(sigma_t^2) sum_k lam_k f(k)^2.
    mean_square_e = np.mean(sig2) * np.sum(lam * f**2)
    return np.sqrt(mean_square_e) / Sp
