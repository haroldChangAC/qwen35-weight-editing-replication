import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, networkx as nx
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from data import S, F4
from data2 import *
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","TeX Gyre Termes"],"font.size":8,
  "axes.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"pdf.fonttype":42})
SC, DC = 3.3, 6.8
def save(fig, name):
    fig.savefig(f"{name}.png", bbox_inches="tight"); fig.savefig(f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
SHORT = {"base":"Original","base_rp105":"RP 1.05","base_rp110":"RP 1.10","base_rp120":"RP 1.20","p10c":"p10c","p25c":"p25c",
         "p50c":"p50c","p50c_a050":"α = 0.5","p50c_a150":"α = 1.5","p50c_rp110":"p50c + RP 1.10"}
OSTY = {"correct":("0.15",None,"Correct"),"wrong":("0.55",None,"Wrong answer"),"no_answer":("0.80","////","Finished, no answer"),
        "cut_noloop":("white","....","Hit limit, no loop"),"cut_loop":("0.90",None,"Hit limit, ends in loop")}

# Fig 5 (exact quantiles, 2% steps)
fig, ax = plt.subplots(figsize=(SC, 2.4))
for lab, k, ls, c in [("Original","base","-","0.0"),("Original + RP 1.10","base_rp110","--","0.35"),("Edit p50c","p50c","-","0.55"),("p50c + RP 1.10","p50c_rp110",":","0.0")]:
    fin = 100*(1-S[k][3]); pts = [(t/1000, p) for t, p in zip(Q[k], QP) if t < 32768]
    ax.plot([0]+[x for x,_ in pts]+[32.768], [0]+[y for _,y in pts]+[fin], ls=ls, color=c, lw=1.1, label=f"{lab}: {fin:.0f}% finish")
ax.axvline(32.768, color="0.7", lw=0.6); ax.text(32.3, 97, "limit", ha="right", va="top", fontsize=6.6, color="0.3")
ax.set_xlabel("Generated tokens (thousands)"); ax.set_ylabel("Answers finished within length (%)")
ax.set_xlim(0, 34); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=6.4, loc="upper left")
save(fig, "fig5_length_cdf")

# Fig 6: alluvial, Original -> p50c -> p50c + RP 1.10
stages = ["base","p50c","p50c_rp110"]; titles = ["Original","Edit p50c","p50c + RP 1.10"]
gap, W, X = 9, 0.16, [0, 1, 2]
def blocks(counts):
    y, out = 500 + gap*4, []
    for n in counts: out.append((y-n, y)); y -= n + gap
    return out
B = [blocks(F4[s]) for s in stages]
fig, ax = plt.subplots(figsize=(DC, 3.4)); ax.axis("off")
for si in range(2):
    m = TRANS[(stages[si], stages[si+1])]
    lcur = [b[1] for b in B[si]]; rcur = [b[1] for b in B[si+1]]
    for i in range(5):
        for j in range(5):
            n = m[i][j]
            if not n: continue
            ly1, ly0 = lcur[i], lcur[i]-n; ry1, ry0 = rcur[j], rcur[j]-n; lcur[i] -= n; rcur[j] -= n
            x0, x1 = X[si]+W/2, X[si+1]-W/2; xm = (x0+x1)/2
            verts = [(x0,ly1),(xm,ly1),(xm,ry1),(x1,ry1),(x1,ry0),(xm,ry0),(xm,ly0),(x0,ly0),(x0,ly1)]
            codes = [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY]
            dest = O[j]; shade = {"correct":"0.25","wrong":"0.55","no_answer":"0.7","cut_noloop":"0.82","cut_loop":"0.9"}[dest]
            ax.add_patch(PathPatch(Path(verts, codes), fc=shade, ec="none", alpha=0.55 if dest != "correct" else 0.75))
for si, s in enumerate(stages):
    for o, (y0, y1), n in zip(O, B[si], F4[s]):
        if n == 0: continue
        fc, h, _ = OSTY[o]
        ax.add_patch(Rectangle((X[si]-W/2, y0), W, y1-y0, fc=fc, hatch=h, ec="black", lw=0.5))
        if n >= 18: ax.text(X[si], (y0+y1)/2, str(n), ha="center", va="center", fontsize=7, color="white" if o in ("correct","wrong") else "black",
                            bbox=None if o in ("correct","wrong") else dict(fc="white", ec="none", pad=0.6))
    ax.text(X[si], 500+gap*4+14, titles[si], ha="center", va="bottom", fontsize=8.5, weight="bold")
ax.set_xlim(-0.35, 2.35); ax.set_ylim(-5, 560)
hs = [Rectangle((0,0),1,1,fc=OSTY[o][0],hatch=OSTY[o][1],ec="black",lw=0.4) for o in O]
ax.legend(hs, [OSTY[o][2] for o in O], ncol=5, frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 0.0))
save(fig, "fig6_alluvial")

# Fig 7: (a) co-solving network, (b) coverage matrix
n = np.array([OVL[i][i] for i in range(10)]); J = np.array([[OVL[i][j]/(n[i]+n[j]-OVL[i][j]) for j in range(10)] for i in range(10)])
fam = {c: ("decoding" if c.startswith("base") else "authors" if c in ("p10c","p25c","p50c") else "ours") for c in C}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(DC, 3.3), gridspec_kw={"width_ratios":[1, 1.08]})
G = nx.Graph()
for i, c in enumerate(C): G.add_node(c)
for i in range(10):
    for j in range(i+1, 10):
        if J[i,j] >= 0.25: G.add_edge(C[i], C[j], w=J[i,j])
ang = np.linspace(90, 90 - 360, 11)[:-1] * np.pi / 180
pos = {c: (np.cos(t), np.sin(t)) for c, t in zip(C, ang)}
for u, v, d in sorted(G.edges(data=True), key=lambda e: e[2]["w"]):
    a1.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.2", lw=0.3+5.0*(d["w"]-0.25), alpha=0.18+0.75*(d["w"]-0.25)/0.45, zorder=1, solid_capstyle="round")
mk = {"decoding":"o","authors":"s","ours":"D"}
for i, c in enumerate(C):
    x, y = pos[c]
    a1.scatter(x, y, s=25+1.6*n[i], marker=mk[fam[c]], fc="white" if fam[c]=="decoding" else "0.6" if fam[c]=="ours" else "0.25", ec="black", lw=0.6, zorder=3)
    r = 1.33 + 0.0011*n[i]
    a1.text(r*x, r*y, f"{SHORT[c]}\n({n[i]})", ha="center", va="center", fontsize=6.4, zorder=4)
top = sorted(G.edges(data=True), key=lambda e: -e[2]["w"])[:3]
for u, v, d in top:
    mx, my = 0.42*pos[u][0]+0.58*pos[v][0], 0.42*pos[u][1]+0.58*pos[v][1]
    a1.text(mx, my, f"{d['w']:.2f}", fontsize=6, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)
a1.set_xlim(-1.85, 1.85); a1.set_ylim(-1.75, 1.75); a1.set_aspect("equal")
a1.set_title("(a) Co-solving network", fontsize=8.5, loc="left"); a1.axis("off")
lh = [plt.Line2D([], [], marker=mk[f], ls="", mfc=fc, mec="black", ms=5) for f, fc in (("decoding","white"),("authors","0.25"),("ours","0.6"))]
a1.legend(lh, ["Decoding only","Authors' edits","Our variants"], frameon=False, fontsize=6.4, loc="lower left", bbox_to_anchor=(-0.06, -0.16))
a1.text(0.56, -0.08, "Node size = problems solved.\nEdge = Jaccard overlap ≥ 0.25;\nwider and darker = larger;\nthree largest labelled.", transform=a1.transAxes, fontsize=6.2, color="0.25", va="top")
cov = np.array([[100*OVL[i][j]/n[i] for j in range(10)] for i in range(10)])
a2.imshow(cov, cmap="Greys", vmin=0, vmax=100)
for i in range(10):
    for j in range(10):
        a2.text(j, i, f"{cov[i,j]:.0f}", ha="center", va="center", fontsize=5.6, color="white" if cov[i,j] > 60 else "black")
a2.set_xticks(range(10)); a2.set_yticks(range(10)); a2.set_xticklabels([SHORT[c] for c in C], rotation=55, ha="right", fontsize=6.2)
a2.set_yticklabels([f"{SHORT[c]} ({n[i]})" for i, c in enumerate(C)], fontsize=6.2)
a2.set_xlabel("…also solved by B (%)", fontsize=7); a2.set_ylabel("Problems solved by A", fontsize=7)
a2.set_title("(b) Coverage: share of A's solved problems also solved by B", fontsize=8, loc="left")
for sp in a2.spines.values(): sp.set_visible(False)
a2.tick_params(length=0)
fig.tight_layout(w_pad=1.5); save(fig, "fig7_cosolving")

# Fig 8: accuracy by difficulty level and by subject
fig, (b1, b2) = plt.subplots(1, 2, figsize=(DC, 3.0), gridspec_kw={"width_ratios":[5, 7.6]})
lv = np.array([[100*LEVEL[c][k]/NLEVEL[k] for k in range(5)] for c in C])
sb = np.array([[100*SUBJACC[c][k]/NSUBJ[k] for k in range(7)] for c in C])
for ax, M, xl, title in ((b1, lv, [f"L{k+1}\n(n={NLEVEL[k]})" for k in range(5)], "(a) By difficulty level"),
                         (b2, sb, [f"{s}\n(n={m})" for s, m in zip(["Algebra","Count. &\nProb.","Geometry","Interm.\nAlgebra","Number\nTheory","Pre-\nalgebra","Pre-\ncalculus"], NSUBJ)], "(b) By subject")):
    ax.imshow(M, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=6.2, color="white" if M[i,j] > 55 else "black")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(xl, fontsize=5.8); ax.xaxis.tick_top()
    ax.set_yticks(range(10)); ax.set_yticklabels([SHORT[c] for c in C] if ax is b1 else [], fontsize=6.4)
    ax.set_title(title, fontsize=8.5, loc="left", pad=26)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    for yb in (3.5, 6.5): ax.axhline(yb, color="white", lw=2)
fig.text(0.5, -0.02, "Cell = MATH-500 accuracy (%) within that slice; each row is one condition (n = 500 problems in total).", ha="center", fontsize=6.6, color="0.25")
fig.tight_layout(w_pad=0.8); save(fig, "fig8_level_subject")
print("FIGS2_OK", "J(p50c,a150)=%.2f" % J[6,8], "J(rp110,p50c)=%.2f" % J[2,6])
