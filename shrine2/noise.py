"""The profile-noise variance sigma_t^2 and the structure parameter uncertainty sigma_S.

Paper: Section 3.3.2 (sigma_t^2) and Section 3.1 (Eq. sigS).
"""
import numpy as np
from scipy.fft import dct, idct
from scipy.ndimage import uniform_filter1d


def noise_variance(x, kc, n_avg=15):
    """sigma_t^2 at every sample of one profile x (Section 3.3.2).

    1. Keep only the DCT modes k > k_min, where k_min = max(N/4, 2 k_c), at most 3N/4. These lie above the
       burst's power, so the reconstruction holds only noise.
    2. Average its square over n_avg = 15 samples centred on t (end samples repeated at the edges).
    3. Divide by 1 - k_min/N: white noise keeps only that fraction of its variance in those modes.

    Args:
        x: one profile, shape (N,) (in dm_uncertainty, the profile at d_max).
        kc: filter cutoff k_c, which sets k_min.
        n_avg: number of samples in the running mean square.
    Returns:
        sigma_t^2 at every sample, shape (N,), in the units of x squared.
    """
    N = x.size
    k_min = max(N // 4, min(2 * kc, 3 * N // 4))
    c = dct(x, norm="ortho")
    c[:k_min] = 0                                    # remove modes k = 1..k_min
    n_hi = idct(c, norm="ortho")                     # noise-only profile
    return uniform_filter1d(n_hi**2, size=n_avg, mode="nearest") / (1 - k_min / N)


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
    """Typical length of the filtered noise relative to the structure parameter, sqrt(<||e||^2>) / Sp.

    The expansion behind the uncertainties needs ||e|| << ||a|| (Eq. small); Section 5.2 discusses where it holds
    (the simulations span ~0.3 at S/N 15 to ~0.07 at S/N 200). For noise uncorrelated between samples,
    <||e||^2> = trace(D1 S diag(sigma_t^2) S D1^T) ~ mean(sigma_t^2) sum_k lam_k f(k)^2 (exact for constant
    sigma_t^2). This is a diagnostic only: it is not used in sigma_S or sigma_DM.

    Args:
        sig2: sigma_t^2, shape (N,).
        kc: filter cutoff k_c.
        Sp: the structure parameter at d_max.
    Returns:
        sqrt(<||e||^2>) / Sp, a float.
    """
    from .structure import filter_response
    f, lam = filter_response(sig2.size, kc)
    return np.sqrt(np.mean(sig2) * np.sum(lam * f**2)) / Sp
