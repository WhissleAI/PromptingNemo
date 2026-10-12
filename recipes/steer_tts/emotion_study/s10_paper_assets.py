"""paper_stats.json -> paper assets: results/cremad.json and figures/F_cremad.pdf (+ .png preview).

Runs in the paper environment (matplotlib + the paper's scripts/style.py):

    python s10_paper_assets.py <stats.json> <paper_dir>

Figure (ACL column width, 3.03 in):
  (a) heatmap of d (mean parallel delta / pooled within-actor neutral SD) per acted emotion x the 12
      SteerTTS controls; cells whose 95% actor-bootstrap CI excludes 0 are annotated.
  (b) median word-level peak-intensity delta vs the same actor's neutral word: first vs last word.
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

stats_path, paper = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(paper, "scripts"))
from style import PALETTE, apply, figsize  # noqa: E402

CONTROLS = ["rate", "pause", "loud", "f0_med", "f0_iqr", "f0_end", "tilt", "voiced", "flat", "centroid", "cpps", "h1h2"]
EMO = ["ANG", "HAP", "FEA", "SAD", "DIS"]
rows = json.load(open(stats_path))
shutil.copy(stats_path, os.path.join(paper, "results", "cremad.json"))
get = lambda m, **kw: [r for r in rows if r["metric"] == m and all(r[k] == v for k, v in kw.items())]

apply()
fig, (ax, bx) = plt.subplots(2, 1, figsize=figsize("column", 1.02), gridspec_kw=dict(height_ratios=[1.0, 0.62], hspace=0.62))

# ---- (a) heatmap
M = np.full((len(EMO), len(CONTROLS)), np.nan)
sig = np.zeros_like(M, bool)
for i, e in enumerate(EMO):
    for j, c in enumerate(CONTROLS):
        r = get("steer_delta_d", emotion=e, control=c)[0]
        M[i, j] = r["value"]
        sig[i, j] = r["ci_lo"] is not None and (r["ci_lo"] > 0 or r["ci_hi"] < 0)
# diverging: blue (decrease) -> light grey midpoint -> vermillion (increase)
cmap = LinearSegmentedColormap.from_list("div", [PALETTE["blue"], "#f2f2f2", PALETTE["vermillion"]])
lim = 2.0
im = ax.imshow(np.clip(M, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(range(len(CONTROLS)))
ax.set_xticklabels(CONTROLS, rotation=60, ha="right",
                   rotation_mode="anchor", fontsize=6.5, family="sans-serif")
ax.set_yticks(range(len(EMO)))
ax.set_yticklabels(EMO, fontsize=7)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(CONTROLS)), minor=True)
ax.set_yticks(np.arange(-0.5, len(EMO)), minor=True)
ax.grid(which="minor", color="white", lw=1.0)
ax.tick_params(which="minor", length=0)
for i in range(len(EMO)):
    for j in range(len(CONTROLS)):
        if sig[i, j] and abs(M[i, j]) >= 0.2:
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}".replace("-", "−"), ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 1.3 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-2, -1, 0, 1, 2])
cb.ax.tick_params(labelsize=6.5, length=1.5)
cb.outline.set_visible(False)
cb.set_label(r"$d$ vs. same-actor neutral", fontsize=6.5)
ax.set_title("(a) parallel deltas in control units", fontsize=7.5, loc="left")

# ---- (b) first vs last word intensity
x = np.arange(len(EMO))
w = 0.36
for k, (pos, col) in enumerate([("first word", PALETTE["sky"]), ("last word", PALETTE["blue"])]):
    vals, lo, hi = [], [], []
    for e in EMO:
        r = get("word_delta_median", emotion=e, control="dInt_dB", split=pos)[0]
        vals.append(r["value"]); lo.append(r["value"] - r["ci_lo"]); hi.append(r["ci_hi"] - r["value"])
    bx.bar(x + (k - 0.5) * w, vals, w * 0.92, color=col, label=pos, yerr=[lo, hi],
           error_kw=dict(lw=0.6, capsize=1.2, ecolor=PALETTE["grey"]))
bx.axhline(0, color=PALETTE["black"], lw=0.5)
bx.set_xticks(x)
bx.set_xticklabels(EMO, fontsize=7)
bx.set_ylabel(r"$\Delta$ peak level (dB)", fontsize=7)
bx.grid(axis="x", visible=False)
bx.legend(loc="upper right", ncol=2, handlelength=1.0, columnspacing=0.8, fontsize=6.5)
bx.set_title("(b) word-level loudness delta, first vs. last word", fontsize=7.5, loc="left")

out = os.path.join(paper, "figures", "F_cremad.pdf")
fig.savefig(out)
fig.savefig(os.path.join(os.path.dirname(stats_path), "F_cremad.png"), dpi=220)
print("wrote", out, "and results/cremad.json")
