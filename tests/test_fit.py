"""The fit window and the parabola fit (Section 3.3.3, Eqs. parab, lsq)."""
import numpy as np

from shrine2.fit import fit_parabola, fit_window, parabola_least_squares


def test_parabola_recovers_vertex_and_curvature():
    d = np.linspace(-1, 1.5, 41)
    Sp = 5 - 3 * (d - 0.2) ** 2
    d_fit, c, B = fit_parabola(d, Sp)
    assert np.isclose(d_fit, 0.2)
    assert np.allclose(c, [5, 0, -3], atol=1e-10)


def test_B_inverts_A():
    x = np.linspace(-1, 1, 11)
    B, _ = parabola_least_squares(x, x)
    A = np.column_stack([np.ones_like(x), x, x**2])
    assert np.allclose(B @ A, np.eye(3))


def test_window_is_contiguous_about_the_peak():
    # Level 3: indices 4-6 (index 8 is beyond the dip).
    Sp = np.array([0, 1, 3, 2.5, 3.2, 4, 3.6, 2, 3.9, 1, 0.0])
    win = fit_window(Sp, 1.0, int(np.argmax(Sp)))
    assert np.flatnonzero(win).tolist() == [4, 5, 6]


def test_window_at_the_edge_returns_none():
    assert fit_window(np.array([5.0, 4.9, 3, 1]), 1.0, 0) is None
