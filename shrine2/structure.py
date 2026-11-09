"""The structure parameter, the coefficients w and the filter cutoff k_c.

Paper: Section 2.2 (Eqs. smooth, sp), Section 3.1 (Eq. weights), Section 3.3.1 (Eq. ahat) and
Appendix 1 (Eqs. Mk, runmean).

Everything is done in the DCT basis. With c = C^T x, the orthonormal type-II DCT of a profile x:

    S x       ->  f(k) c_k                         the low-pass filter (Eq. smooth)
    D1^T D1   ->  lam_k = 2 - 2 cos[(k-1) pi/N]     D1^T D1 is diagonal in the DCT-II basis

so ||D1 S x||^2 = sum_k lam_k f(k)^2 c_k^2, and no N x N matrices are needed.
"""
import numpy as np
from scipy.fft import dct, idct

# Filter order O (Eq. smooth), as in S23.
O = 3


def filter_response(N, kc):
    """f(k) and lam_k for modes k = 1..N.

    f(k)  = 1 / (1 + (k/k_c)^(2O))     low-pass response (Eq. smooth)
    lam_k = 2 - 2 cos[(k-1) pi/N]      eigenvalues of D1^T D1

    Args:
        N: number of samples in a profile, which is also the number of DCT modes.
        kc: filter cutoff k_c.
    Returns:
        f: f(k) for k = 1..N, shape (N,).
        lam: lam_k for k = 1..N, shape (N,).
    """
    k = np.arange(1, N + 1)
    f = 1 / (1 + (k / kc) ** (2 * O))
    lam = 2 - 2 * np.cos((k - 1) * np.pi / N)
    return f, lam


def smooth(I, kc):
    """S x (Eq. smooth), for one profile or for every row of I(d_j, t).

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at
            trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        S x, the same shape as I.
    """
    f, _ = filter_response(I.shape[-1], kc)

    # The filter multiplies each DCT coefficient by f(k); transform back to samples.
    c = dct(I, norm="ortho", axis=-1)
    return idct(f * c, norm="ortho", axis=-1)


def structure_parameter(I, kc):
    """Sp = ||D1 S x|| (Eq. sp), for one profile or for every row of I(d_j, t).

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at
            trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        Sp: one number per profile (a float for one profile, shape (n_d,) for I(d_j, t)), in the
        units of x.
    """
    f, lam = filter_response(I.shape[-1], kc)
    c = dct(I, norm="ortho", axis=-1)

    # ||D1 S x||^2 = sum_k lam_k f(k)^2 c_k^2, because S and D1^T D1 are both diagonal here.
    Sp_squared = np.sum(lam * f**2 * c**2, axis=-1)
    return np.sqrt(Sp_squared)


def coefficients(I, kc):
    """w(t) = (D1 S x)^T D1 S / Sp (Eq. ahat), for one profile or for every row of I(d_j, t).

    This is Eq. weights, w = a_hat^T D1 S, with a_hat estimated from the data as D1 S x / Sp.
    As a column, w = S D1^T D1 S x / Sp, which in the DCT basis is IDCT(lam f^2 c) / Sp.
    e_par = sum_t w_t n_t is then the part of the noise that changes Sp at first order
    (Eq. weights).

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at
            trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        w: the same shape as I; row j is w(t; d_j).
    """
    f, lam = filter_response(I.shape[-1], kc)
    c = dct(I, norm="ortho", axis=-1)

    # S D1^T D1 S x in the DCT basis is lam_k f(k)^2 c_k; transform it back to samples.
    w_times_Sp = idct(lam * f**2 * c, norm="ortho", axis=-1)

    # Divide each profile's row by its own Sp.
    Sp = structure_parameter(I, kc)
    return w_times_Sp / np.expand_dims(Sp, -1)


def choose_kc(I, extend=True, verbose=False):
    """k_c from S23's procedure (Appendix 1), with the paper's extended search unless extend=False.

    M(k)    = largest |DCT coefficient k| over the trial DMs                 (Eq. Mk)
    Mbar(k) = mean of M over the six modes ending at k                       (Eq. runmean)
    M_n     = mean of M over the upper half of the modes                     (the noise floor)

    S23: k_c is the first mode where Mbar falls to M_n.
    Extended search: if Mbar rises above M_n + 4 sig_Mbar again between k_c and 2 k_c
    (sig_Mbar = spread of Mbar over the upper half), the spectrum has risen above the noise again;
    k_c then moves to the first mode after the last such rise at which Mbar falls back to M_n.

    As in SHRINE's dm_processing.get_kc, k_c is a 0-based array index, used as it is in f(k) with
    k = 1..N, so it is one mode below the 1-based mode at which Mbar falls to the floor. The k_c
    values in S23, SHRINE and the paper all follow this convention, so it is kept.

    Args:
        I: I(d_j, t), shape (n_d, N), over the trial DMs (the wider the range, the better M(k) is
            sampled).
        extend: True for the paper's extended search (Appendix 1); False for S23's procedure alone.
        verbose: print the noise floor and where each step puts k_c.
    Returns:
        k_c, an int.
    Raises:
        RuntimeError if Mbar never falls to the noise floor, or (extended search) never falls back
        to it after rising above the noise again.
    """
    # M(k): the largest |DCT coefficient| of each mode over the trial DMs (Eq. Mk).
    M = np.max(np.abs(dct(I, norm="ortho", axis=-1)), axis=0)
    N = M.size

    # Mbar(k): the running mean of M over the six modes ending at k (Eq. runmean).
    # The first N values of the full convolution are Mbar[i] = mean(M[i-5..i]).
    Mbar = np.convolve(M, np.ones(6) / 6)[:N]

    # M_n: the noise floor, the mean of M over the upper half of the modes.
    Mn = np.mean(M[N // 2:])

    # S23: the first mode at which Mbar falls to the noise floor, starting at the first full
    # six-mode window (index 5).
    kc = None
    for i in range(5, N):
        if Mbar[i] <= Mn:
            kc = i
            break
    if kc is None:
        raise RuntimeError("could not find k_c: the DCT spectrum never falls to the noise floor")
    if verbose:
        print(f"  k_c, S23 (App. 1, Eqs. Mk, runmean): Mbar falls to the noise floor "
              f"M_n = {Mn:.4g} at k_c = {kc}")
    if not extend:
        return kc

    # Extended search: the level M_n + 4 sig_Mbar, with sig_Mbar the spread of Mbar over the upper
    # half, from six modes above N/2 (as in the paper's code).
    rise_level = Mn + 4 * np.std(Mbar[N // 2 + 6:])

    # The last mode between k_c and 2 k_c (and below N/2) at which Mbar is above that level.
    last_rise = None
    for i in range(kc, min(2 * kc, N // 2)):
        if Mbar[i] > rise_level:
            last_rise = i
    if last_rise is None:
        if verbose:
            print(f"  k_c, extended search (App. 1): Mbar stays below M_n + 4 sig_Mbar = "
                  f"{rise_level:.4g} between k_c and 2 k_c, so k_c = {kc}")
        return kc

    # The spectrum rose above the noise again: k_c moves to the first mode from the last rise on at
    # which Mbar falls back to the noise floor.
    for i in range(last_rise, N):
        if Mbar[i] <= Mn:
            if verbose:
                print(f"  k_c, extended search (App. 1): Mbar rises above M_n + 4 sig_Mbar = "
                      f"{rise_level:.4g} again at mode {last_rise}, so k_c moves from {kc} to {i}")
            return i
    raise RuntimeError("could not find k_c: after rising above the noise again at mode "
                       f"{last_rise}, the DCT spectrum never falls back to the noise floor")
