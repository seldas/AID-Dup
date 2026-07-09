#!/usr/bin/env python3
"""
Figure 1. Pair-level F1 by deduplication strategy across 12 FAERS benchmark series.

Standalone script: no input files are required. All values are hardcoded from the
analysis dataset used in the manuscript. The script writes one 300-dpi JPG file
next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_FILE = Path(__file__).with_suffix(".jpg")
DPI = 300

# Hardcoded pair-level F1 values by benchmark series.
series = [f"SE{i}" for i in range(1, 13)]
ether_baseline = [
    0.7269, 0.9211, 0.9057, 0.1679, 0.2368, 0.8919,
    0.0000, 0.8190, 0.9775, 0.0870, 0.6154, 0.9884,
]
ai_enhanced_dedup = [
    0.7264, 0.9410, 0.9165, 0.5500, 0.8056, 0.9143,
    0.0000, 0.9053, 0.9775, 0.2500, 1.0000, 0.9825,
]
llm_first_pipeline = [
    0.5123, 0.9316, 0.8307, 0.1205, 0.5455, 0.9167,
    0.0000, 0.9533, 0.9412, 0.1622, 0.6154, 0.9221,
]

# Colorblind-aware colors that match the manuscript's broad palette.
colors = {
    "ETHER-based baseline": "#4C78A8",
    "AID-Dup": "#F58518",
    "LLM-first pipeline": "#54A24B",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    }
)

x = np.arange(len(series))
width = 0.25

fig, ax = plt.subplots(figsize=(7.2, 4.1))

ax.bar(
    x - width,
    ether_baseline,
    width,
    label="ETHER-based baseline",
    color=colors["ETHER-based baseline"],
    edgecolor="white",
    linewidth=0.4,
)
ax.bar(
    x,
    ai_enhanced_dedup,
    width,
    label="AID-Dup",
    color=colors["AID-Dup"],
    edgecolor="white",
    linewidth=0.4,
)
ax.bar(
    x + width,
    llm_first_pipeline,
    width,
    label="LLM-first pipeline",
    color=colors["LLM-first pipeline"],
    edgecolor="white",
    linewidth=0.4,
)

ax.set_title(
    "Pair-level duplicate-detection performance across 12 human-adjudicated FAERS benchmark series",
    pad=10,
)
ax.set_xlabel("Benchmark case series")
ax.set_ylabel("Pair-level F1")
ax.set_xticks(x)
ax.set_xticklabels(series)
ax.set_ylim(0, 1.08)
ax.set_xlim(-0.65, len(series) - 0.35)
ax.yaxis.grid(True, linestyle="-", linewidth=0.35, alpha=0.35)
ax.set_axisbelow(True)

# Mark the negative-control series without obscuring the bars.
se7_index = series.index("SE7")
ax.annotate(
    "negative-control series",
    xy=(se7_index, 0.015),
    xytext=(se7_index, 1.035),
    ha="center",
    va="top",
    fontsize=7.5,
    color="#555555",
    arrowprops={"arrowstyle": "-[,widthB=1.0", "lw": 0.8, "color": "#777777"},
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.6,
)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, format="jpg", dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 95})
plt.close(fig)
print(f"Wrote {OUTPUT_FILE}")
