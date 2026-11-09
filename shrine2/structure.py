"""The structure parameter, the coefficients w and the filter cutoff k_c.

Paper: Section 2.2 (Eqs. smooth, sp), Section 3.1 (Eq. weights), Section 3.3.1 (Eq. ahat) and Appendix 1
(Eqs. Mk, runmean).

Everything is done in the DCT basis. With c = C^T x, the orthonormal type-II DCT of a profile x:

    S x       ->  f(k) c_k                       the low-pass filter (Eq. smooth)
    D1^T D1   ->  lam_k = 2 - 2 cos[(k-1) pi/N]   D1^T D1 is diagonal in the DCT-II basis

so ||D1 S x||^2 = sum_k lam_k f(k)^2 c_k^2, and no N x N matrices are needed.
"""
import numpy as np
from scipy.fft import dct, idct

O = 3   # filter order (Eq. smooth), as in S23


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
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        S x, the same shape as I.
    """
    f, _ = filter_response(I.shape[-1], kc)
    return idct(f * dct(I, norm="ortho", axis=-1), norm="ortho", axis=-1)


def structure_parameter(I, kc):
    """Sp = ||D1 S x|| (Eq. sp), for one profile or for every row of I(d_j, t).

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        Sp: one number per profile (a float for one profile, shape (n_d,) for I(d_j, t)), in the units of x.
    """
    f, lam = filter_response(I.shape[-1], kc)
    c = dct(I, norm="ortho", axis=-1)
    return np.sqrt(np.sum(lam * f**2 * c**2, axis=-1))


def coefficients(I, kc):
    """w(t) = (D1 S x)^T D1 S / Sp (Eq. ahat), for one profile or for every row of I(d_j, t).

    This is Eq. weights, w = a_hat^T D1 S, with a_hat estimated from the data as D1 S x / Sp.
    As a column, w = S D1^T D1 S x / Sp, which in the DCT basis is IDCT(lam f^2 c) / Sp.
    e_par = sum_t w_t n_t is then the part of the noise that changes Sp at first order (Eq. weights).

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        w: the same shape as I; row j is w(t; d_j).
    """
    f, lam = filter_response(I.shape[-1], kc)
    c = dct(I, norm="ortho", axis=-1)
    Sp = np.sqrt(np.sum(lam * f**2 * c**2, axis=-1))
    w = idct(lam * f**2 * c, norm="ortho", axis=-1)
    return w / np.expand_dims(Sp, -1)


def choose_kc(I, extend=True):
    """k_c from S23's procedure (Appendix 1), with the paper's extended search unless extend=False.

    M(k)    = largest |DCT coefficient k| over the trial DMs                 (Eq. Mk)
    Mbar(k) = mean of M over the six modes ending at k                       (Eq. runmean)
    M_n     = mean of M over the upper half of the modes                     (the noise floor)

    S23: k_c is the first mode where Mbar falls to M_n.
    Extended search: if Mbar rises above M_n + 4 sig_Mbar again between k_c and 2 k_c (sig_Mbar = spread of
    Mbar over the upper half), the spectrum has risen above the noise again; k_c then moves to the first mode
    after the last such rise at which Mbar falls back to M_n.

    As in SHRINE's dm_processing.get_kc, k_c is a 0-based array index, used as it is in f(k) with k = 1..N.

    Args:
        I: I(d_j, t), shape (n_d, N), over the trial DMs (the wider the range, the better M(k) is sampled).
        extend: True for the paper's extended search (Appendix 1); False for S23's procedure alone.
    Returns:
        k_c, an int.
    Raises:
        RuntimeError if Mbar never falls to the noise floor.
    """
    M = np.max(np.abs(dct(I, norm="ortho", axis=-1)), axis=0)               # Eq. Mk
    N = M.size
    Mbar = np.convolve(M, np.ones(6) / 6)[:N]                               # Eq. runmean: Mbar[i] = mean(M[i-5..i])
    Mn = np.mean(M[N // 2:])                                                # noise floor

    kc = next((i for i in range(5, N) if Mbar[i] <= Mn), None)              # S23
    if kc is None:
        raise RuntimeError("could not find k_c: the DCT spectrum never falls to the noise floor")
    if not extend:
        return kc

    rise = Mn + 4 * np.std(Mbar[N // 2 + 6:])                               # M_n + 4 sig_Mbar
    above = [i for i in range(kc, min(2 * kc, N // 2)) if Mbar[i] > rise]
    if above:                                                               # risen above the noise again
        kc = next((i for i in range(above[-1], N) if Mbar[i] <= Mn), N // 2)
    return kc
