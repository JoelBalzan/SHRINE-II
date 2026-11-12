"""Sp = ||D1 S x|| and w agree with the explicit matrices of the paper (Eqs. smooth, sp, ahat)."""
import numpy as np
from scipy.fft import idct

from shrine2.structure import (choose_kc, coefficients, filter_response,
                               forward_difference_matrix, structure_parameter)

rng = np.random.default_rng(1)
N = 64
kc = 9


def explicit_D1S(N, kc):
    """D1 S as explicit matrices (Eq. smooth and the forward difference D1).

    Args:
        N: number of samples.
        kc: filter cutoff k_c.
    Returns:
        D1 S, shape (N-1, N).
    """
    # Columns: the DCT-II basis vectors, so x = C c.
    C = idct(np.eye(N), norm="ortho", axis=0)
    f = filter_response(N, kc)
    S = C @ np.diag(f) @ C.T

    # (N-1) x N forward differences.
    D1 = np.diff(np.eye(N), axis=0)
    return D1 @ S


def test_forward_difference_matrix():
    """The sparse D1 is the (N-1) x N forward-difference matrix."""
    assert np.array_equal(forward_difference_matrix(N).toarray(), np.diff(np.eye(N), axis=0))


def test_structure_parameter_matches_matrices():
    x = rng.normal(size=N) + np.exp(-0.5 * ((np.arange(N) - 30) / 4) ** 2) * 10
    assert np.isclose(structure_parameter(x, kc), np.linalg.norm(explicit_D1S(N, kc) @ x))


def test_coefficients_match_matrices():
    """w = (D1 S x)^T D1 S / Sp (Eq. ahat), with D1 S as an explicit matrix."""
    # Its own random numbers, so that the other tests draw the same ones as before.
    x = np.random.default_rng(2).normal(size=N) + np.exp(-0.5 * ((np.arange(N) - 30) / 4) ** 2) * 10
    D1S = explicit_D1S(N, kc)
    D1Sx = D1S @ x
    w = D1Sx @ D1S / np.linalg.norm(D1Sx)
    assert np.allclose(coefficients(x, kc), w, rtol=0, atol=1e-14)


def test_coefficients_match_finite_differences():
    """w = d Sp / d x (Eq. ahat): the change in Sp from a small change in each sample."""
    x = rng.normal(size=N) + np.exp(-0.5 * ((np.arange(N) - 30) / 4) ** 2) * 10
    h = 1e-6

    # Central differences of Sp with respect to each sample.
    grad = []
    for e in np.eye(N):
        Sp_plus = structure_parameter(x + h * e, kc)
        Sp_minus = structure_parameter(x - h * e, kc)
        grad.append((Sp_plus - Sp_minus) / (2 * h))
    grad = np.array(grad)

    assert np.allclose(coefficients(x, kc), grad, atol=1e-7)


def spectrum_rows(M, n_dm=5):
    """Profiles whose largest |DCT coefficient| per mode is M (plus a little scatter between rows).

    Args:
        M: the largest |DCT coefficient| of each mode, shape (N,).
        n_dm: number of profiles (trial DMs).
    Returns:
        profiles, shape (n_dm, N); row 0 has DCT coefficients exactly M.
    """
    c = M[None, :] * (0.9 + 0.1 * rng.random((n_dm, M.size)))
    c[0] = M
    return idct(c, norm="ortho", axis=-1)


def test_kc_single_scale_is_unchanged_by_the_extended_search():
    M = np.concatenate([np.full(20, 10.0), 1 + 0.05 * rng.random(236)])
    I = spectrum_rows(M)
    assert choose_kc(I, extend=False) == choose_kc(I, extend=True)


def test_kc_extended_search_passes_a_dip():
    """Burst power at k < 15 and again at 22-35, with a dip into the noise between (Appendix 1,
    Figure 10b)."""
    noise = 1 + 0.05 * rng.random(256)
    M = noise.copy()
    M[:15] = 10
    M[22:36] = 5
    I = spectrum_rows(M)

    k_s23 = choose_kc(I, extend=False)
    k_ext = choose_kc(I, extend=True)
    assert 15 <= k_s23 < 22
    assert k_ext >= 36
