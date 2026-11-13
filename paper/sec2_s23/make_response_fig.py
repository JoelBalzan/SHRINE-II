"""Figure 5 (fig:s23_response, Section 2.4): the spectral responses of S23's smoothing, f(k) = 1 / (1 + (k/k_c)^{2O}) with O = 3, the residual's 1 - f(k),
and the smoothed residual's f(k)[1 - f(k)], against k / k_c. On a k / k_c axis, since
the curves depend only on k / k_c and O.

Writes figures/fig_s23_response.pdf.

usage: python make_response_fig.py
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from plotting import IBM_PALETTE, savefig, set_pub_col, set_pub_style
set_pub_col(1); set_pub_style(use_latex=False)
plt.rcParams["figure.constrained_layout.use"] = False
BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE
O = 3
u = np.linspace(0, 2.5, 600); f = 1 / (1 + u ** (2 * O))
fig, ax = plt.subplots(figsize=(3.4, 1.85))
fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.25)
ax.plot(u, f, color="k", lw=1.1); ax.plot(u, 1 - f, color="0.5", lw=1.1, ls="--"); ax.plot(u, f * (1 - f), color=PURP, lw=1.3)
ax.axvline(1, color="k", lw=0.7, ls=":")
ax.text(0.04, 0.9, r"$f(k)$: smoothing", fontsize=6.5, ha="left", va="top")
ax.text(2.46, 0.86, r"$1-f(k)$: residual", fontsize=6.5, ha="right", va="top", color="0.4")
ax.text(1.47, 0.47, "$f(k)\\,[1-f(k)]$:\nsmoothed residual", fontsize=6.5, ha="left", va="top", color=PURP, linespacing=1.2)
ax.set_xlim(0, 2.5); ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xlabel(r"$k/k_c$"); ax.set_ylabel("Response"); ax.grid(True, lw=0.5, alpha=0.5)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
savefig(os.path.join(HERE, "figures", "fig_s23_response.pdf"), dpi=600, fig=fig)
