"""Figure 2 (fig:s23_vec, Section 2.3): schematic of the structure vector and its noise. Not to scale;
noise exaggerated.
(a) S23: D1 S x = D1 S x_tilde + D1 S Delta_x, with the triangle-inequality bounds ||D1 S x_tilde|| +- ||D1 S Delta_x||.
    D1 S x is the same vector as in (b), but its parts are drawn with different lengths from a and e.
(b) This work: D1 S x = a + e, e = e_par a_hat + e_perp; S ~ ||a|| + e_par + ||e_perp||^2 / (2 ||a||).

Writes figures/fig_s23_vec.pdf.

usage: python make_vec_fig.py
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from plotting import IBM_PALETTE, savefig, set_pub_col, set_pub_style
set_pub_col(1); set_pub_style(use_latex=False)
plt.rcParams["figure.constrained_layout.use"] = False
BLUE, PURP, MAG, ORNG, YELL = IBM_PALETTE


def arrow(ax, p, q, color, lw=1.4, ls="-", z=3):
    ax.annotate("", xy=q, xytext=p, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=9, shrinkA=0, shrinkB=0))


def dim(ax, x1, y, label, color):
    """Dimension line from the origin to x1 at height y, label above it, dotted extension line up to the axis."""
    ax.annotate("", xy=(x1, y), xytext=(0, y), zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
    for xx in (0, x1): ax.plot([xx, xx], [-0.03, y - 0.06], color=color, lw=0.5, ls=":", zorder=1)
    ax.text(x1 / 2, y + 0.03, label, ha="center", va="bottom", fontsize=6.5, color=color)


WBOX = dict(fc="white", ec="none", pad=0.4)
Y1, Y2 = -1.12, -1.47
fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.75))
fig.subplots_adjust(left=0.03, right=0.97, top=0.955, bottom=0.01, hspace=0.12)
O = np.array([0.0, 0.0])

# (a) S23's split. D1 S x is the same vector as in (b), so the orange arrow has the same length,
# but its two parts differ from a and e: D1 S Delta_x is shorter than e (as for the example
# burst, where it is about 0.6 ||e||) and nearly perpendicular to D1 S x_tilde, whose length
# follows from |x_tilde + residual| = |a + e|.
ax = axs[0]
S_len = np.linalg.norm([2.6 + 0.45, 0.85])                 # ||a + e|| of panel (b)
r = 0.6; ang = np.deg2rad(95)                              # ||D1 S Delta_x|| and its direction
L = -r * np.cos(ang) + np.sqrt(S_len**2 - (r * np.sin(ang))**2)   # ||D1 S x_tilde||
xt = np.array([L, 0.0])
res = r * np.array([np.cos(ang), np.sin(ang)]); x = xt + res
ax.add_patch(Circle(xt, r, fill=False, color="0.6", ls="--", lw=0.7, zorder=1))
arrow(ax, O, xt, "k"); arrow(ax, xt, x, MAG); arrow(ax, O, x, ORNG)
ax.text(L / 2, -0.12, r"$\mathbf{D}_1\mathbf{S}\tilde{\mathbf{x}}$", ha="center", va="top", fontsize=8)
ax.text(*(xt + 0.5 * res + np.array([0.18, 0.06])), r"$\mathbf{D}_1\mathbf{S}\boldsymbol{\Delta}_{\mathbf{x}}$", color=MAG, fontsize=8, ha="left", va="center", bbox=WBOX)
ax.text(*(0.5 * x + np.array([-0.15, 0.17])), r"$\mathbf{D}_1\mathbf{S}\mathbf{x}$", color=ORNG, fontsize=8, ha="right", va="bottom")
dim(ax, L - r, Y1, r"$\|\mathbf{D}_1\mathbf{S}\tilde{\mathbf{x}}\|-\|\mathbf{D}_1\mathbf{S}\boldsymbol{\Delta}_{\mathbf{x}}\|$", "0.45")
dim(ax, L + r, Y2, r"$\|\mathbf{D}_1\mathbf{S}\tilde{\mathbf{x}}\|+\|\mathbf{D}_1\mathbf{S}\boldsymbol{\Delta}_{\mathbf{x}}\|$", "0.45")
ax.set_title("(a) S23: signal estimate and residual", loc="left", fontsize=8.5)

# (b) this work
ax = axs[1]
a = np.array([2.6, 0.0]); epar, eperp = 0.45, 0.85
e = np.array([epar, eperp]); x = a + e; S = np.linalg.norm(x); ne = np.linalg.norm(e)
ax.add_patch(Circle(a, ne, fill=False, color="0.6", ls="--", lw=0.7, zorder=1))
arrow(ax, O, a, "k"); arrow(ax, a, a + np.array([epar, 0]), BLUE); arrow(ax, a + np.array([epar, 0]), x, YELL)
arrow(ax, a, x, "0.35", lw=0.9, ls=(0, (3, 2))); arrow(ax, O, x, ORNG)
ax.annotate("", xy=(S, 0), xytext=(a[0] + epar, 0), zorder=4,       # second-order lengthening, ||e_perp||^2 / 2||a||, as an arrow
            arrowprops=dict(arrowstyle="-|>", color=ORNG, lw=1.4, mutation_scale=6, shrinkA=0, shrinkB=0))
ax.text(1.0, -0.12, r"$\mathbf{a}$", ha="center", va="top", fontsize=8.5)
ax.text(a[0] + epar / 2, -0.12, r"$e_\parallel\hat{\mathbf{a}}$", color=BLUE, ha="center", va="top", fontsize=8)
ax.text(S + 0.07, eperp * 0.5, r"$\mathbf{e}_\perp$", color="#b07800", ha="left", va="center", fontsize=8, bbox=WBOX)
ax.text(a[0] + 0.12, eperp * 0.62, r"$\mathbf{e}$", color="0.35", ha="right", va="center", fontsize=8)
ax.text(*(0.5 * x + np.array([-0.15, 0.17])), r"$\mathbf{D}_1\mathbf{S}\mathbf{x}=\mathbf{a}+\mathbf{e}$", color=ORNG, fontsize=8, ha="right", va="bottom")
ax.text(S + 0.04, 0.0, r"$\simeq\|\mathbf{e}_\perp\|^2/2\|\mathbf{a}\|$", color=ORNG, fontsize=6.5, ha="left", va="center", bbox=WBOX, zorder=5)  # level with the arrow
dim(ax, 2.6 - ne, Y1, r"$\|\mathbf{a}\|-\|\mathbf{e}\|$", "0.45")
dim(ax, 2.6 + ne, Y2, r"$\|\mathbf{a}\|+\|\mathbf{e}\|$", "0.45")
ax.set_title(r"(b) This work: structure vector and noise", loc="left", fontsize=8.5)

for ax in axs:
    ax.plot(*O, marker="o", ms=2.5, color="k"); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.15, 3.95); ax.set_ylim(-1.55, 1.08)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
savefig(os.path.join(HERE, "figures", "fig_s23_vec.pdf"), dpi=600, fig=fig)
