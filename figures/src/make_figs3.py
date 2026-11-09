import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from data3 import *
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes"], "font.size": 8,
  "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})
SC, DC = 3.3, 6.8
def save(fig, name):
    fig.savefig(f"{name}.png", bbox_inches="tight"); fig.savefig(f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

M = lambda k: np.array([[ALL[c][k] for c in row] for row in G]) * 100   # rows = alpha, cols = RP
ACC, LO, HI, CUT, LOOP = M(0), M(1), M(2), M(3), M(4)

def upsample(Z, f=40):
    """Bilinear interpolation between the measured cells, used only for contour lines and the 3-D surface."""
    r, c = Z.shape; yi = np.linspace(0, r - 1, (r - 1) * f + 1); xi = np.linspace(0, c - 1, (c - 1) * f + 1)
    Zr = np.array([np.interp(xi, np.arange(c), row) for row in Z])
    return xi, yi, np.array([np.interp(yi, np.arange(r), col) for col in Zr.T]).T

halo = [pe.withStroke(linewidth=1.8, foreground="white")]
def heat(ax, Z, title, vmax, levels, ci=None):
    # shading and contours are bilinear between the 12 measured cells; boxed numbers are the measured values
    xi, yi, Zi = upsample(Z)
    Zp = np.pad(Zi, 20, mode="edge")  # edge cells shaded flat out to the cell border
    ax.imshow(Zp, cmap="Greys", vmin=-0.05 * vmax, vmax=1.25 * vmax, aspect="auto", extent=[-0.5, 2.5, 3.5, -0.5],
              interpolation="nearest")
    cs = ax.contour(xi, yi, Zi, levels=levels, colors="0.0", linewidths=0.7, linestyles="-")
    C = np.array([(j, a) for a in range(4) for j in range(3)], float); pos = []
    for segs in cs.allsegs:   # put each label at the point of its line farthest from every boxed value
        V = np.concatenate([v for v in segs if len(v)]) if any(len(v) for v in segs) else None
        if V is None: continue
        d = np.min(np.hypot((V[:, None, 0] - C[None, :, 0]) * 1.6, V[:, None, 1] - C[None, :, 1]), axis=1)
        pos.append(tuple(V[np.argmax(d)]))
    for t in ax.clabel(cs, fmt="%d", fontsize=5.8, inline=True, inline_spacing=3, manual=pos): t.set_path_effects(halo)
    for a in range(4):
        for j in range(3):
            lab = f"{Z[a, j]:.1f}" + (f"\n[{ci[0][a, j]:.0f}, {ci[1][a, j]:.0f}]" if ci is not None else "")
            ax.text(j, a, lab, ha="center", va="center", fontsize=6.6, linespacing=1.0, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.3", lw=0.5))
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{r:.2f}" for r in RP])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{a:g}" for a in ALPHA])
    ax.set_xlim(-0.35, 2.35); ax.set_ylim(3.35, -0.35)
    ax.set_xlabel("Repetition penalty"); ax.set_title(title, fontsize=8, loc="left")
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0)

# Fig 9: the alpha x penalty grid
fig = plt.figure(figsize=(DC, 2.9))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.5)
ax = fig.add_subplot(gs[0]); heat(ax, ACC, "(a) Accuracy (%)", 60, [10, 20, 30, 40, 50])
ax.set_ylabel("Edit strength α (× p50c diff)")
heat(fig.add_subplot(gs[1]), CUT, "(b) Hit length limit (%)", 100, [30, 50, 70, 90])
heat(fig.add_subplot(gs[2]), LOOP, "(c) Ends in a loop (%)", 100, [10, 30, 50, 70, 90])
ax = fig.add_subplot(gs[3])
ax.axhline(0, color="0.6", lw=0.6)
for j, (mk, ls, fc, lab, dx) in {1: ("o", "-", "white", "RP 1.05 vs 1.00", -0.025), 2: ("s", "--", "0.0", "RP 1.10 vs 1.00", 0.025)}.items():
    d = np.array([DRP[(a, j)] for a in range(4)]) * 100; x = np.array(ALPHA) + dx
    ax.errorbar(x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], fmt=mk + ls, color="0.0", mfc=fc, ms=4,
                lw=0.9, capsize=2, elinewidth=0.7, label=lab)
ax.text(1.6, 19, "penalty helps", ha="right", va="center", fontsize=6.4, color="0.35", style="italic")
ax.text(1.6, -18, "penalty hurts", ha="right", va="center", fontsize=6.4, color="0.35", style="italic")
ax.set_xticks(ALPHA); ax.set_xlim(-0.15, 1.65); ax.set_ylim(-20, 22)
ax.set_xlabel("Edit strength α"); ax.set_ylabel("Δ accuracy (pp)")
ax.set_title("(d) Penalty effect by α (95% CI)", fontsize=8, loc="left")
ax.legend(frameon=False, fontsize=6.2, loc="upper right", bbox_to_anchor=(1.02, 0.9))
save(fig, "fig9_alpha_penalty_grid")

# Fig S1: 3-D surfaces (bilinear between the 12 measured cells; dots = measured)
fig = plt.figure(figsize=(DC, 3.0))
for k, (Z, title, zl, zmax) in enumerate([(ACC, "(a) Accuracy", "Accuracy (%)", 60), (LOOP, "(b) Ends in a loop", "Loop share (%)", 100)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    xi, yi, Zi = upsample(Z, 12)
    X, Y = np.meshgrid(1 + 0.05 * xi, 0.5 * yi)
    ax.plot_surface(X, Y, Zi, cmap="Greys", vmin=-0.15 * zmax, vmax=1.1 * zmax, rstride=2, cstride=2,
                    edgecolor="0.35", linewidth=0.25, alpha=0.9, antialiased=True)
    XX, YY = np.meshgrid(RP, ALPHA)
    ax.scatter(XX.ravel(), YY.ravel(), Z.ravel(), color="black", s=9, depthshade=False)
    ax.contour(X, Y, Zi, levels=6, zdir="z", offset=0, colors="0.45", linewidths=0.5)
    ax.set_xticks(RP); ax.set_xticklabels([f"{r:.2f}" for r in RP], fontsize=6.4)
    ax.set_yticks(ALPHA); ax.set_yticklabels([f"{a:g}" for a in ALPHA], fontsize=6.4)
    ax.tick_params(axis="z", labelsize=6.4, pad=1)
    ax.set_xlabel("Repetition penalty", fontsize=7, labelpad=2); ax.set_ylabel("Edit strength α", fontsize=7, labelpad=2)
    ax.set_zlabel(zl, fontsize=7, labelpad=1); ax.set_zlim(0, zmax)
    ax.view_init(elev=24, azim=-128 if k == 0 else -52)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis): p.pane.set_facecolor((1, 1, 1, 0)); p.pane.set_edgecolor("0.8")
    ax.set_title(title, fontsize=8, loc="left", y=0.98)
fig.subplots_adjust(left=0.0, right=0.97, wspace=-0.05)
save(fig, "figS1_alpha_penalty_surface")
print("done")
