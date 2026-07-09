#!/usr/bin/env python3
"""
Figure 3. Model sensitivity by deduplication architecture.

Standalone script: no input files are required. All values are hardcoded from the
analysis dataset used in the manuscript. The script writes one 300-dpi JPG file
next to this script. The single output file contains both panels.

Design intent:
- Panel A avoids using taller bars for the less desirable LLM-first/scratch
  behavior. Instead, it uses a horizontal dumbbell plot of model-dependent F1
  range. Rightward movement means greater model sensitivity / less robustness.
- Panel B shows the macro-F1 trajectory by model for each architecture, matching
  the manuscript text and Table 4.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_FILE = Path(__file__).with_suffix(".jpg")
DPI = 300

series = [f"SE{i}" for i in range(1, 13)]
models = ["Llama-3.1", "Llama-4", "Claude Haiku 4.5", "Claude Sonnet 4.6"]

# Pair-level F1 for each model and benchmark series.
# Values are hardcoded to make the script fully self-contained.
enhanced = {
    "Llama-3.1": [0.5992, 0.9286, 0.8300, 0.5405, 0.8056, 0.9143, 0.0000, 0.8958, 0.9775, 0.1714, 0.6667, 0.9515],
    "Llama-4": [0.6520, 0.9296, 0.9165, 0.5238, 0.8056, 0.8986, 0.0000, 0.8776, 0.9775, 0.2000, 1.0000, 0.9853],
    "Claude Haiku 4.5": [0.7408, 0.9392, 0.9200, 0.5500, 0.8056, 0.8986, 0.0000, 0.9053, 0.9775, 0.2500, 1.0000, 0.9547],
    "Claude Sonnet 4.6": [0.7264, 0.9410, 0.9165, 0.5500, 0.8056, 0.9143, 0.0000, 0.9053, 0.9775, 0.2500, 1.0000, 0.9825],
}

llm_first = {
    "Llama-3.1": [0.0893, 0.3529, 0.2824, 0.0185, 0.0352, 0.6545, 0.0000, 0.1802, 0.6466, 0.0058, 0.0000, 0.1196],
    "Llama-4": [0.2495, 0.9052, 0.8274, 0.0525, 0.3824, 0.6545, 0.0000, 0.7692, 0.7361, 0.0164, 0.5714, 0.9221],
    "Claude Haiku 4.5": [0.4317, 0.9108, 0.8184, 0.1149, 0.5455, 0.9167, 0.0000, 0.9533, 0.9412, 0.0643, 0.7273, 0.9221],
    "Claude Sonnet 4.6": [0.5123, 0.9316, 0.8307, 0.1205, 0.5455, 0.9167, 0.0000, 0.9533, 0.9412, 0.1622, 0.6154, 0.9221],
}

# Colors are fixed for manuscript consistency across figures.
COLOR_ENHANCED = "#2E7D32"
COLOR_LLM_FIRST = "#E67E22"
COLOR_CONNECTOR = "#BDBDBD"
COLOR_GRID = "#D9D9D9"
COLOR_TEXT = "#222222"
COLOR_NOTE = "#666666"
COLOR_REFERENCE = "#4D4D4D"

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


def to_matrix(data):
    """Return a models x series numpy array."""
    return np.array([data[model] for model in models], dtype=float)


def compute_summary(data):
    matrix = to_matrix(data)
    series_range = matrix.max(axis=0) - matrix.min(axis=0)
    macro_by_model = matrix.mean(axis=1)
    return {
        "matrix": matrix,
        "macro_by_model": macro_by_model,
        "series_range": series_range,
        "series_mean": matrix.mean(axis=0),
        "macro_spread": macro_by_model.max() - macro_by_model.min(),
        "mean_series_range": series_range.mean(),
    }


enh = compute_summary(enhanced)
llm = compute_summary(llm_first)
range_gap = llm["series_range"] - enh["series_range"]

fig = plt.figure(figsize=(7.4, 5.9))
grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.55, 1.0], hspace=0.43)

# Panel A: series-level sensitivity as a horizontal dumbbell plot.
# This avoids implying that taller LLM-first bars represent better performance.
ax1 = fig.add_subplot(grid[0, 0])
y = np.arange(len(series))

for yi, x_enh, x_llm in zip(y, enh["series_range"], llm["series_range"]):
    ax1.plot(
        [x_enh, x_llm],
        [yi, yi],
        color=COLOR_CONNECTOR,
        linewidth=1.3,
        alpha=0.95,
        zorder=1,
    )

ax1.scatter(
    enh["series_range"],
    y,
    s=38,
    color=COLOR_ENHANCED,
    edgecolor="white",
    linewidth=0.6,
    label=f"AID-Dup, mean range={enh['mean_series_range']:.3f}",
    zorder=3,
)
ax1.scatter(
    llm["series_range"],
    y,
    s=42,
    color=COLOR_LLM_FIRST,
    edgecolor="white",
    linewidth=0.6,
    label=f"LLM-first pipeline, mean range={llm['mean_series_range']:.3f}",
    zorder=4,
)

# Mean model-sensitivity reference lines, distinct from the data marks.
ax1.axvline(enh["mean_series_range"], color=COLOR_ENHANCED, linestyle="--", linewidth=1.0, alpha=0.65, zorder=0)
ax1.axvline(llm["mean_series_range"], color=COLOR_LLM_FIRST, linestyle="--", linewidth=1.0, alpha=0.65, zorder=0)

# Highlight the largest excess sensitivities of the LLM-first/scratch approach.
top_gap_idx = np.argsort(range_gap)[-3:]
for idx in top_gap_idx:
    ax1.text(
        llm["series_range"][idx] + 0.018,
        idx,
        f"+{range_gap[idx]:.2f}",
        ha="left",
        va="center",
        fontsize=7,
        color=COLOR_NOTE,
    )

# Mark SE7 as a degenerate negative-control series where all models have F1=0.
se7_idx = series.index("SE7")
ax1.text(
    0.022,
    se7_idx + 0.28,
    "SE7: all models F1=0",
    ha="left",
    va="center",
    color=COLOR_NOTE,
    fontsize=7,
)

# Directional cue makes clear that rightward position is undesirable sensitivity.
ax1.annotate(
    "more model-dependent / less robust",
    xy=(0.78, -0.95),
    xytext=(0.30, -0.95),
    arrowprops=dict(arrowstyle="->", color=COLOR_REFERENCE, lw=0.9),
    ha="left",
    va="center",
    fontsize=8,
    color=COLOR_REFERENCE,
)

ax1.set_title("A. Series-level model sensitivity", loc="left", pad=6, fontweight="bold", color=COLOR_TEXT)
ax1.set_xlabel("F1 range across four LLMs (max-min within series; lower is more stable)")
ax1.set_ylabel("Benchmark case series")
ax1.set_xlim(-0.02, 0.88)
ax1.set_ylim(len(series) - 0.45, -1.25)
ax1.set_yticks(y)
ax1.set_yticklabels(series)
ax1.grid(True, axis="x", color=COLOR_GRID, linestyle="-", linewidth=0.6, alpha=0.75, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, ncol=1, loc="lower right", bbox_to_anchor=(1.0, 0.02), handlelength=1.4)

# Panel B: macro-F1 trajectory by model, showing architecture-level robustness.
ax2 = fig.add_subplot(grid[1, 0])
model_x = np.arange(len(models))
ax2.plot(
    model_x,
    enh["macro_by_model"],
    color=COLOR_ENHANCED,
    marker="o",
    linewidth=2.2,
    markersize=5.5,
    label=f"AID-Dup (spread={enh['macro_spread']:.3f})",
)
ax2.plot(
    model_x,
    llm["macro_by_model"],
    color=COLOR_LLM_FIRST,
    marker="o",
    linewidth=2.2,
    markersize=5.5,
    label=f"LLM-first pipeline (spread={llm['macro_spread']:.3f})",
)

# Annotate the two end points to emphasize stability vs dependence on model strength.
ax2.annotate(
    "stable across\nmodel tiers",
    xy=(2.75, enh["macro_by_model"][-1]),
    xytext=(2.15, 0.82),
    arrowprops=dict(arrowstyle="->", color=COLOR_ENHANCED, lw=0.9),
    color=COLOR_ENHANCED,
    fontsize=8,
    ha="left",
    va="center",
)
ax2.annotate(
    "large gain only with\nstronger models",
    xy=(3.0, llm["macro_by_model"][-1]),
    xytext=(1.75, 0.39),
    arrowprops=dict(arrowstyle="->", color=COLOR_LLM_FIRST, lw=0.9),
    color=COLOR_LLM_FIRST,
    fontsize=8,
    ha="left",
    va="center",
)

for i, val in enumerate(enh["macro_by_model"]):
    ax2.text(i, val + 0.026, f"{val:.3f}", ha="center", va="bottom", fontsize=7, color=COLOR_ENHANCED)
for i, val in enumerate(llm["macro_by_model"]):
    ax2.text(i, val - 0.042, f"{val:.3f}", ha="center", va="top", fontsize=7, color=COLOR_LLM_FIRST)

ax2.set_title("B. Macro-F1 trajectory by model", loc="left", pad=6, fontweight="bold", color=COLOR_TEXT)
ax2.set_ylabel("Macro-F1 across 12 series")
ax2.set_ylim(0.10, 0.82)
ax2.set_xlim(-0.25, len(models) - 0.75)
ax2.set_xticks(model_x)
ax2.set_xticklabels(models, rotation=0)
ax2.grid(True, axis="y", color=COLOR_GRID, linestyle="-", linewidth=0.6, alpha=0.75)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc="lower right")

for ax in (ax1, ax2):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)

fig.savefig(OUTPUT_FILE, format="jpg", dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 95})
plt.close(fig)
print(f"Wrote {OUTPUT_FILE}")
