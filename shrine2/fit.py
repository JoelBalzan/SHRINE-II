"""The fit window and the parabola fit.

Paper: Section 3.3.3 (Eqs. parab, match, dvertex) and Section 3.3.4 (Eq. lsq).
"""
import numpy as np


def fit_window(Sp, sig_S, j_max):
    """The fit window (Section 3.3.3): S23's interval (Eq. s23int) with sigma_S in place of their uncertainty.

    That is the contiguous trial DMs about the peak with Sp(d_j) >= Sp_max - sigma_S, found by stepping out from
    the peak one trial DM at a time on each side.

    Args:
        Sp: Sp(d_j) at every trial DM, shape (n_d,).
        sig_S: sigma_S at d_max.
        j_max: index of d_max in Sp.
    Returns:
        win: boolean array, shape (n_d,), True for the trial DMs in the window; or None if the window reaches
        either end of the trial DMs (then widen the DM range).
    """
    level = Sp[j_max] - sig_S
    lo = hi = j_max
    while lo > 0 and Sp[lo - 1] >= level:
        lo -= 1
    while hi < Sp.size - 1 and Sp[hi + 1] >= level:
        hi += 1
    if lo == 0 or hi == Sp.size - 1:
        return None
    win = np.zeros(Sp.size, bool)
    win[lo:hi + 1] = True
    return win


def least_squares(x, y):
    """B = (A^T A)^-1 A^T and c = B y for a parabola in x (Eq. lsq).

    Row j of A holds the numbers that multiply c0, c1, c2 at x_j: (1, x_j, x_j^2).
    pinv(A) equals (A^T A)^-1 A^T here (A has full column rank) and is numerically stabler.

    Args:
        x: trial DMs relative to the parabola's origin, shape (n,).
        y: Sp at those trial DMs, shape (n,).
    Returns:
        B: (A^T A)^-1 A^T, shape (3, n).
        c: (c0, c1, c2) = B y, shape (3,).
    """
    A = np.column_stack([np.ones_like(x), x, x**2])
    B = np.linalg.pinv(A)
    return B, B @ y


def parabola(d, Sp):
    """Least-squares parabola Sp(d) ~ c0 + c1 (d - d_fit) + c2 (d - d_fit)^2 over the trial DMs d (Eq. parab).

    Two passes:
      1. fit about the centre of the window (for numerical conditioning only) to locate the vertex, d_fit;
      2. refit the same parabola about d_fit, so that c1 = 0 and row 1 of B, B[1], gives c1 about the vertex,
         whose scatter between noise realisations moves the vertex by -c1/(2 c2) (Eq. dvertex).

    Args:
        d: the trial DMs in the fit window, shape (n,), pc cm^-3.
        Sp: Sp at those trial DMs, shape (n,).
    Returns:
        (d_fit, c, B): the vertex (pc cm^-3); c = (c0, c1, c2) about d_fit, shape (3,); B, shape (3, n),
        with c_m = sum_j B_mj Sp(d_j). None if the parabola has no maximum (c2 >= 0).
    """
    d_mid = d.mean()
    _, c = least_squares(d - d_mid, Sp)                  # pass 1
    if c[2] >= 0:
        return None
    d_fit = d_mid - c[1] / (2 * c[2])                    # vertex
    B, c = least_squares(d - d_fit, Sp)                  # pass 2: about the vertex (c[1] ~ 0)
    return d_fit, c, B
