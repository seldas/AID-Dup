"""
Paired bootstrap comparison of macro-F1 between two deduplication conditions
across the benchmark series. No numpy dependency -- pure Python, deterministic
given a seed, so results are reproducible and traceable in a manuscript.

Motivation: every headline number in this study (e.g. "AI-enhanced Dedup
0.747 vs ETHER-based baseline 0.612") is a single-run point estimate over
12 series with no uncertainty measure. This resamples SERIES (not
LLM-call noise -- that's a separate, unaddressed source of variance; see
the manuscript's Limitations) to give a bootstrap 95% CI on the macro-F1
difference and a bootstrap p-value, so "is this just 12 lucky/unlucky
data points" has an actual answer.
"""

import random


def bootstrap_macro_f1_comparison(
    f1_a: list, f1_b: list, n_boot: int = 10000, seed: int = 42, ci: float = 0.95
) -> dict:
    """Paired bootstrap over matched per-series F1 values for two conditions
    (f1_a[i] and f1_b[i] must be the same series, in the same order).

    Resamples series indices WITH replacement n_boot times; for each
    resample, recomputes each condition's macro-F1 (mean of the resampled
    per-series F1s) and the delta (a - b). Returns point estimates,
    percentile confidence intervals, and a two-sided bootstrap p-value for
    delta != 0 (doubled smaller-tail proportion, capped at 1.0).
    """
    n = len(f1_a)
    if n != len(f1_b):
        raise ValueError("f1_a and f1_b must be the same length (paired per series)")
    if n < 2:
        raise ValueError("Need at least 2 matched series to bootstrap")

    rng = random.Random(seed)
    macro_a = sum(f1_a) / n
    macro_b = sum(f1_b) / n
    point_delta = macro_a - macro_b

    boot_a = []
    boot_b = []
    boot_delta = []
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in range(n)]
        ma = sum(f1_a[i] for i in sample) / n
        mb = sum(f1_b[i] for i in sample) / n
        boot_a.append(ma)
        boot_b.append(mb)
        boot_delta.append(ma - mb)

    boot_a.sort()
    boot_b.sort()
    boot_delta.sort()

    alpha = 1 - ci
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(int((1 - alpha / 2) * n_boot), n_boot - 1)

    def pct(sorted_vals):
        return sorted_vals[lo_idx], sorted_vals[hi_idx]

    delta_lo, delta_hi = pct(boot_delta)
    a_lo, a_hi = pct(boot_a)
    b_lo, b_hi = pct(boot_b)

    n_le0 = sum(1 for d in boot_delta if d <= 0)
    n_ge0 = sum(1 for d in boot_delta if d >= 0)
    p_value = min(1.0, 2 * min(n_le0, n_ge0) / n_boot)

    return {
        "n_series": n,
        "n_boot": n_boot,
        "seed": seed,
        "ci_level": ci,
        "macro_f1_a": round(macro_a, 4),
        "macro_f1_a_ci": [round(a_lo, 4), round(a_hi, 4)],
        "macro_f1_b": round(macro_b, 4),
        "macro_f1_b_ci": [round(b_lo, 4), round(b_hi, 4)],
        "delta": round(point_delta, 4),
        "delta_ci": [round(delta_lo, 4), round(delta_hi, 4)],
        "p_value": round(p_value, 5),
        "significant": delta_lo > 0 or delta_hi < 0,
    }
