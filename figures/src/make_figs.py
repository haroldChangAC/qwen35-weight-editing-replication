import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from data import S, OUT, F4, P, F5
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","TeX Gyre Termes"],"font.size":8,
  "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
  "savefig.dpi":300,"pdf.fonttype":42})
SC, DC = 3.3, 6.8
def save(fig, name):
    fig.savefig(f"{name}.png", bbox_inches="tight"); fig.savefig(f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
pct = lambda v: 100*v

# Fig 1: pipeline
fig, ax = plt.subplots(figsize=(DC, 1.9)); ax.set_xlim(-1.5, 101.5); ax.set_ylim(0, 30); ax.axis("off")
steps = [("Public weights", "Qwen3.5-0.8B\nrev. 2fc0636\ntext-only export"),
         ("Checkpoints", "authors' exact diffs\np10c, p25c, p50c\nSHA-256 verified;\np50c direction ×0.5, ×1.5"),
         ("Generation", "vLLM 0.33.0\nthinking on, T=0.6\ntop-p 0.95, top-k 20\npresence penalty 1.5\n± repetition penalty\nmax 32,768 tokens"),
         ("Grading", "authors' prompt\nand grader\nlast \\boxed{} answer\nmath-verify 0.9.0\nloop rule on the\nlast 200 words"),
         ("Statistics", "MATH-500, n=500\nbootstrap 95% CI\nexact McNemar test\nper problem")]
w, gap = 17.6, 2.9
for i, (h, b) in enumerate(steps):
    x = i*(w+gap)
    ax.add_patch(FancyBboxPatch((x, 2), w, 24, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec="black", lw=0.7))
    ax.text(x+w/2, 23, h, ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(x+w/2, 12.5, b, ha="center", va="center", fontsize=5.9, linespacing=1.35)
    if i < len(steps)-1:
        ax.annotate("", xy=(x+w+gap-0.3, 14), xytext=(x+w+0.3, 14), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))
save(fig, "fig1_pipeline")

# Fig 2: repetition-penalty sweep vs the weight edit
rp = [("base",1.00),("base_rp105",1.05),("base_rp110",1.10),("base_rp120",1.20)]
fig, ax = plt.subplots(figsize=(SC, 2.4))
xs = [x for _, x in rp]; acc = [pct(S[k][0]) for k, _ in rp]
lo = [pct(S[k][0]-S[k][1]) for k, _ in rp]; hi = [pct(S[k][2]-S[k][0]) for k, _ in rp]
e = S["p50c"]; ax.axhspan(pct(e[1]), pct(e[2]), color="0.88", lw=0); ax.axhline(pct(e[0]), color="0.35", ls="--", lw=0.8)
ax.text(1.0, pct(e[0])+2.2, f"Weight edit p50c, no penalty: {pct(e[0]):.1f}%", fontsize=7, color="0.2")
ax.errorbar(xs, acc, yerr=[lo, hi], fmt="-o", color="black", ms=3.5, lw=0.9, capsize=2, elinewidth=0.7)
for x, a in zip(xs, acc): ax.text(x+0.008, a+(-3.0 if x < 1.08 else 0.8), f"{a:.1f}", ha="left", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(["1.00\n(none)", "1.05", "1.10", "1.20"])
ax.set_xlabel("Repetition penalty on original weights"); ax.set_ylabel("MATH-500 accuracy (%)"); ax.set_ylim(0, 60)
save(fig, "fig2_penalty_sweep")

# Fig 3: dose-response in edit scale alpha
al = [("base",0),("p50c_a050",0.5),("p50c",1.0),("p50c_a150",1.5)]
fig, ax = plt.subplots(figsize=(SC, 2.5)); xs = [a for _, a in al]
series = [("Accuracy", 0, "-", "o", "black"), ("Hit 32k limit", 3, "--", "s", "0.35"),
          ("Ends in loop", 4, ":", "^", "0.35"), ("Finished, no answer", 6, "-.", "D", "0.55")]
for name, j, ls, mk, c in series:
    ys = [pct(S[k][j]) for k, _ in al]
    if j == 0:
        ax.errorbar(xs, ys, yerr=[[pct(S[k][0]-S[k][1]) for k, _ in al], [pct(S[k][2]-S[k][0]) for k, _ in al]],
                    fmt=ls+mk, color=c, ms=3.5, lw=1.0, capsize=2, elinewidth=0.7, label=name)
    else:
        ax.plot(xs, ys, ls=ls, marker=mk, color=c, ms=3.2, lw=0.9, mfc="white", label=name)
ax.axvline(1.0, color="0.75", lw=0.6, zorder=0)
ax.set_xticks(xs); ax.set_xticklabels(["0\n(original)", "0.5", "1.0\n(p50c)", "1.5"])
ax.set_xlabel("Scale α of the p50c edit direction"); ax.set_ylabel("Share of problems (%)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
save(fig, "fig3_dose_response")

# Fig 4: outcome composition per condition
order = [("Original","base"),("Original + RP 1.05","base_rp105"),("Original + RP 1.10","base_rp110"),("Original + RP 1.20","base_rp120"),
         ("Edit p10c","p10c"),("Edit p25c","p25c"),("Edit p50c","p50c"),
         ("p50c direction ×0.5","p50c_a050"),("p50c direction ×1.5","p50c_a150"),("p50c + RP 1.10","p50c_rp110")]
style = {"correct":("0.15",None,"Correct"), "wrong":("0.55",None,"Wrong answer"), "no_answer":("0.80","////","Finished, no answer"),
         "cut_noloop":("white","....","Hit limit, no loop"), "cut_loop":("0.90",None,"Hit limit, ends in loop")}
fig, ax = plt.subplots(figsize=(DC, 3.0)); ys = list(range(len(order)))[::-1]
for y, (lab, k) in zip(ys, order):
    left = 0
    for o, n in zip(OUT, F4[k]):
        fc, hatch, _ = style[o]; v = n/5
        ax.barh(y, v, left=left, color=fc, hatch=hatch, edgecolor="black", lw=0.4, height=0.68)
        if o == "correct" and v >= 4: ax.text(left+v/2, y, f"{v:.1f}", ha="center", va="center", color="white", fontsize=6.8)
        left += v
ax.set_yticks(ys); ax.set_yticklabels([l for l, _ in order]); ax.set_xlim(0, 100); ax.set_xlabel("Share of 500 MATH-500 problems (%)")
for yb in (5.5, 2.5): ax.axhline(yb, color="0.6", lw=0.5, ls="--")
hs = [plt.Rectangle((0,0),1,1,fc=style[o][0],hatch=style[o][1],ec="black",lw=0.4) for o in OUT]
ax.legend(hs, [style[o][2] for o in OUT], ncol=5, frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.42, 1.0))
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
save(fig, "fig4_outcomes")

# Fig 5: generation length (approximate CDF from 5% quantiles)
fig, ax = plt.subplots(figsize=(SC, 2.4))
cs = [("Original","base","-","0.0"),("Original + RP 1.10","base_rp110","--","0.35"),("Edit p50c","p50c","-","0.55"),("p50c + RP 1.10","p50c_rp110",":","0.0")]
for lab, k, ls, c in cs:
    fin = 100*(1-S[k][3]); pts = [(t/1000, p) for t, p in zip(F5[k], P) if t < 32768]
    xs = [0] + [x for x, _ in pts] + [32.768]; ys = [0] + [y for _, y in pts] + [fin]
    ax.plot(xs, ys, ls=ls, color=c, lw=1.1, label=f"{lab}: {fin:.0f}% finish")
ax.axvline(32.768, color="0.7", lw=0.6); ax.text(32.3, 97, "limit", ha="right", va="top", fontsize=6.6, color="0.3")
ax.set_xlabel("Generated tokens (thousands)"); ax.set_ylabel("Answers finished within length (%)")
ax.set_xlim(0, 34); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=6.4, loc="upper left")
save(fig, "fig5_length_cdf")
print("FIGS_OK")
