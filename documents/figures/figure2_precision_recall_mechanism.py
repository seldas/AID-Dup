#!/usr/bin/env python3
"""
Figure 2. Precision-recall mechanism of the primary performance gain.

Standalone script: no input files are required. All values are hardcoded from the
analysis dataset used in the manuscript. The script writes one 300-dpi JPG file
next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_FILE = Path(__file__).with_suffix(".jpg")
DPI = 300

# Hardcoded macro precision, recall, and F1 across all 12 benchmark series.
# X-axis is recall; Y-axis is precision.
points = {
    "ETHER-based baseline": {"recall": 0.7390, "precision": 0.5801, "f1": 0.6115, "color": "#4C78A8"},
    "AID-Dup": {"recall": 0.7413, "precision": 0.7606, "f1": 0.7474, "color": "#F58518"},
    "LLM-first pipeline": {"recall": 0.6813, "precision": 0.6290, "f1": 0.6210, "color": "#54A24B"},
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    }
)

fig, ax = plt.subplots(figsize=(5.8, 4.8))

# Reference diagonal where precision equals recall.
ax.plot([0.54, 0.80], [0.54, 0.80], linestyle="--", linewidth=1.0, color="#B8B8B8")
ax.text(0.565, 0.585, "precision = recall", rotation=45, fontsize=8, color="#8A8A8A")

# Draw points. Point area scales with macro-F1.
for label, values in points.items():
    ax.scatter(
        values["recall"],
        values["precision"],
        s=450 * values["f1"],
        color=values["color"],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

# Improvement arrow from baseline to AID-Dup.
base = points["ETHER-based baseline"]
enh = points["AID-Dup"]
ax.annotate(
    "",
    xy=(enh["recall"], enh["precision"] - 0.008),
    xytext=(base["recall"], base["precision"] + 0.008),
    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"},
    zorder=2,
)
ax.text(
    0.706,
    0.666,
    "Precision +0.181\nRecall +0.002",
    fontsize=8,
    ha="left",
    va="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.95},
)

# Point labels.
ax.annotate(
    "AID-Dup\nF1=0.747",
    xy=(enh["recall"], enh["precision"]),
    xytext=(enh["recall"] + 0.008, enh["precision"] + 0.006),
    fontsize=8,
    ha="left",
    va="center",
)
ax.annotate(
    "ETHER-based baseline\nF1=0.612",
    xy=(base["recall"], base["precision"]),
    xytext=(base["recall"] - 0.008, base["precision"] - 0.018),
    fontsize=8,
    ha="right",
    va="center",
)
llm = points["LLM-first pipeline"]
ax.annotate(
    "LLM-first pipeline\nF1=0.621",
    xy=(llm["recall"], llm["precision"]),
    xytext=(llm["recall"] + 0.008, llm["precision"] - 0.008),
    fontsize=8,
    ha="left",
    va="center",
)

ax.set_title("LLM adjudication mainly improves precision while preserving recall", pad=10)
ax.set_xlabel("Macro recall across 12 series")
ax.set_ylabel("Macro precision across 12 series")
ax.set_xlim(0.54, 0.80)
ax.set_ylim(0.52, 0.80)
ax.grid(True, linestyle="-", linewidth=0.35, alpha=0.35)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, format="jpg", dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 95})
plt.close(fig)
print(f"Wrote {OUTPUT_FILE}")
