"""Diagnostic plots (written by maximise_structure.py with -s/--save_all, and by maximise_sn.py)."""
import matplotlib

# A non-interactive backend: the plots are only written to files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct

from .structure import smooth

DM_LABEL = r"$\Delta$DM (pc cm$^{-3}$)"


def plot_structure(res, path):
    """Sp(d) with +-sigma_S, the fit window, the parabola and d_fit +- sigma_DM (as Figure 6 of the
    paper).

    Args:
        res: the dict returned by estimate.dm_uncertainty.
        path: output file name.
    """
    d = res["d"]
    Sp = res["Sp"]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Sp(d) with a band of +-sigma_S.
    ax.fill_between(d, Sp - res["sig_S"], Sp + res["sig_S"], color="C1", alpha=0.2, lw=0)
    ax.plot(d, Sp, color="C1", label=r"$\mathcal{S}\pm\sigma_\mathcal{S}$")

    if res["ok"]:
        # The fit window.
        ax.axvspan(res["d_lo"], res["d_hi"], color="C4", alpha=0.1, label="Fit window")

        # The fitted parabola over the window, about its vertex d_fit.
        x = np.linspace(res["d_lo"], res["d_hi"], 200) - res["d_fit"]
        c0, c1, c2 = res["c"]
        ax.plot(x + res["d_fit"], c0 + c1 * x + c2 * x**2, "k--", label="Parabola fit")

        # d_fit +- sigma_DM at the parabola's peak.
        ax.errorbar(res["d_fit"], c0, xerr=res["sig_DM"], fmt="o", color="C4", capsize=3,
                    label=r"$d_{\rm fit}\pm\sigma_{\rm DM}$")

        # Show three window widths about d_fit.
        span = 3 * (res["d_hi"] - res["d_lo"])
        ax.set_xlim(res["d_fit"] - span / 2, res["d_fit"] + span / 2)

    ax.set_xlabel(DM_LABEL)
    ax.set_ylabel(r"Structure parameter $\mathcal{S}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dct(I, kc, path):
    """|DCT coefficients| of every profile against mode number, with k_c (S23's diagnostic plot).

    Args:
        I: I(d_j, t), shape (n_d, N).
        kc: filter cutoff k_c.
        path: output file name.
    """
    modes = np.arange(1, I.shape[1] + 1)
    abs_coefficients = np.abs(dct(I, norm="ortho", axis=-1))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(modes, abs_coefficients.T, ".", ms=1, color="0.5")
    ax.axvline(kc, color="C0", label=f"$k_c$ = {kc}")
    ax.set_xlabel("Mode $k$")
    ax.set_ylabel("|DCT coefficient|")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_profile(x, kc, dt, path):
    """The profile x and the smoothed profile S x at one trial DM.

    Args:
        x: one profile, shape (N,).
        kc: filter cutoff k_c.
        dt: time resolution, us.
        path: output file name.
    """
    t_ms = np.arange(x.size) * dt / 1000

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_ms, x, color="0.6", lw=0.6, label=r"$\mathbf{x}$")
    ax.plot(t_ms, smooth(x, kc), color="C1", label=r"$\mathbf{Sx}$")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("I")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_I_DM_t(I, d, dt, path):
    """I(d, t).

    Args:
        I: I(d_j, t), shape (n_d, N).
        d: the trial DMs, shape (n_d,), uniformly spaced, pc cm^-3.
        dt: time resolution, us.
        path: output file name.
    """
    duration_ms = I.shape[1] * dt / 1000

    # Each row is centred on its trial DM, so the image extends half a step beyond the first and
    # last trial DMs.
    if d.size > 1:
        half_step = (d[1] - d[0]) / 2
    else:
        half_step = 0.5
    d_bottom = d[0] - half_step
    d_top = d[-1] + half_step

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(I, aspect="auto", origin="lower", extent=(0, duration_ms, d_bottom, d_top))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(DM_LABEL)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sn(d, SN, path):
    """Largest boxcar S/N against trial DM (SHRINE's {label}_SN.png).

    Args:
        d: the trial DMs, shape (n_d,), pc cm^-3.
        SN: best boxcar S/N at each, shape (n_d,).
        path: output file name.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d, SN)
    ax.grid(color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel(DM_LABEL)
    ax.set_ylabel("Signal to Noise Ratio")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sn_window(Sx, start, length, dt, path):
    """The smoothed profile at the S/N-maximising DM with its boxcar (SHRINE's
    {label}_Max_SN_window.png).

    Args:
        Sx: the smoothed profile at the S/N-maximising DM, shape (N,).
        start, length: the boxcar, in samples.
        dt: time resolution, us.
        path: output file name.
    """
    t_ms = np.arange(Sx.size) * dt / 1000
    boxcar_start_ms = start * dt / 1000
    boxcar_end_ms = (start + length) * dt / 1000

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_ms, Sx)
    ax.axvspan(boxcar_start_ms, boxcar_end_ms, color="red", alpha=0.3)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("I")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
