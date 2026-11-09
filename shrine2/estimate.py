"""The structure-maximising DM and its 1-sigma uncertainty from one set of profiles I(d_j, t).

Paper: Section 3.3, steps in the order of Sections 3.3.1-3.3.4. Each line below names the equation it evaluates.
"""
import numpy as np

from .covariance import covariance
from .fit import fit_window, parabola
from .noise import noise_ratio, noise_variance, sigma_S
from .structure import coefficients, structure_parameter


def dm_uncertainty(I, d, kc, dt_nu, share=None):
    """Run Section 3.3 on I(d_j, t).

    Args:
        I: I(d_j, t), shape (n_d, N), one row per trial DM.
        d: the trial DMs, shape (n_d,), uniformly spaced, pc cm^-3.
        kc: filter cutoff k_c.
        dt_nu: each channel's dispersion delay per unit DM in samples, shape (n_chan,)
            (profiles.channel_delays divided by dt).
        share: each channel's share of the noise variance, shape (n_chan,); None for equal shares.
    Returns:
        dict with
            ok, why: whether the window and fit succeeded, and why not;
            d, Sp, kc: the inputs and Sp(d_j);
            j_max, d_max, Sp_max: the trial DM with the largest Sp;
            sig_S, sig2, e_ratio: sigma_S, sigma_t^2 and sqrt(<||e||^2>)/Sp at d_max;
            and, when ok:
            win, n_win, d_lo, d_hi: the fit window, its size and its first and last trial DM;
            d_fit, c, B: the vertex, (c0, c1, c2) about it, and B;
            V: the covariance between the trial DMs in the window;
            sig_DM, c2_err: sigma_DM and std[c2]/|c2|.
    """
    d = np.asarray(d, float)
    Sp = structure_parameter(I, kc)                      # Sp(d_j)                              Eq. sp
    j_max = int(np.argmax(Sp))                           # d_max, the trial DM with the largest Sp

    sig2 = noise_variance(I[j_max], kc)                  # sigma_t^2 from the profile at d_max  Sec. 3.3.2
    w_max = coefficients(I[j_max], kc)                   # w at d_max                           Eq. ahat
    sig_S = sigma_S(w_max, sig2)                         # sigma_S                              Eq. sigS

    out = dict(ok=False, why="", d=d, Sp=Sp, kc=kc, j_max=j_max, d_max=d[j_max], Sp_max=Sp[j_max],
               sig_S=sig_S, sig2=sig2, e_ratio=noise_ratio(sig2, kc, Sp[j_max]))

    win = fit_window(Sp, sig_S, j_max)                   # Sp(d_j) >= Sp_max - sigma_S          Sec. 3.3.3
    if win is None:
        out["why"] = "the fit window reaches the end of the DM range: widen the DM range"
        return out
    out["n_win"] = int(win.sum())
    if win.sum() < 3:
        out["why"] = f"only {win.sum()} trial DMs in the fit window: use a finer DM step"
        return out

    fit = parabola(d[win], Sp[win])                      # d_fit, (c0, c1, c2), B               Eqs. parab, lsq
    if fit is None:
        out["why"] = "the parabola over the fit window has no maximum"
        return out
    d_fit, c, B = fit

    step = d[1] - d[0]
    W = coefficients(I[win], kc)                         # w(t; d_j) at every trial DM in the window   Eq. ahat
    V = covariance(W, sig2, dt_nu, step, share)          # V_jl                                 App. 2, Eq. cov

    sig_c1 = np.sqrt(max(B[1] @ V @ B[1], 0))            # std[c1]^2 = sum_jl B_1j B_1l V_jl    Eq. varc1
    sig_DM = sig_c1 / abs(2 * c[2])                      # sigma_DM = std[c1] / |2 c2|          Eq. final
    c2_err = np.sqrt(max(B[2] @ V @ B[2], 0)) / abs(c[2])   # std[c2] / |c2|                    Eq. c2err

    out.update(ok=True, win=win, n_win=int(win.sum()), d_lo=d[win][0], d_hi=d[win][-1],
               d_fit=d_fit, c=c, B=B, V=V, sig_DM=sig_DM, c2_err=c2_err)
    return out


def check_grid(res):
    """Warnings about the trial DMs for a result of dm_uncertainty, and a suggested finer grid.

    The paper's simulations use a step of sigma_DM/6 (35-163 trial DMs in the fit window) and trial DMs out to
    about 15 sigma_DM either side. If the fit window holds fewer than three trial DMs, the suggestion is a step ten
    times finer about d_max. Returns (warnings, grid), with grid = (d_low, d_high, step), or None if the trial DMs
    are fine enough.

    Args:
        res: the dict returned by dm_uncertainty.
    Returns:
        warnings: list of str.
        grid: (d_low, d_high, step) in pc cm^-3 for generate_profiles.py, or None.
    """
    step = res["d"][1] - res["d"][0]
    if not res["ok"]:
        grid = None
        if res.get("n_win", 3) < 3:
            grid = (res["d_max"] - 10 * step, res["d_max"] + 10 * step, step / 10)
        return [res["why"]], grid
    warn, grid = [], None
    sig_DM = res["sig_DM"]
    if step > sig_DM / 3:
        warn.append(f"DM step {step:.3g} is coarser than sigma_DM/3 = {sig_DM / 3:.3g}")
        half = max(15 * sig_DM, 1.5 * (res["d_hi"] - res["d_lo"]))
        grid = (res["d_fit"] - half, res["d_fit"] + half, sig_DM / 6)
    if res["c2_err"] > 0.5:
        warn.append(f"relative curvature error std[c2]/|c2| = {res['c2_err']:.2f} > 0.5: sigma_DM is unreliable "
                    "(the peak may be flat or split; Sec. 3.3.4)")
    if res["e_ratio"] > 0.3:
        warn.append(f"sqrt(<||e||^2>)/Sp = {res['e_ratio']:.2f} > 0.3: the first-order expansion may fail (Sec. 5.2)")
    return warn, grid
