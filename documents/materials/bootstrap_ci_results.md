# Bootstrap confidence intervals for headline macro-F1 comparisons

> Generated 2026-07-08 via the app's `/api/bootstrap-compare` endpoint (see `bootstrap_stats.py`).
> Paired bootstrap over the 12 benchmark series: series are resampled with replacement 10,000 times,
> each condition's macro-F1 is recomputed per resample, and the 95% CI is the 2.5th-97.5th percentile
> of the resampled delta (A - B). Seed=42 for exact reproducibility. Raw per-comparison JSON
> (including every per-series F1 pair used) is in `bootstrap_ci_results.json` next to this file.

**Scope note:** this quantifies series-sampling uncertainty ("is the ranking robust to which 12
series happened to be in the benchmark") -- it does NOT capture LLM output stochasticity
(single-run design per condition; see manuscript Limitations).

| Comparison | A | B | Delta (A-B) | 95% CI | p-value | Significant |
|---|---|---|---|---|---|---|
| ETHER-based baseline vs AID-Dup (Sonnet 4.6) | 0.6115 | 0.7474 | -0.1360 | [-0.2485, -0.0401] | < 0.0001 | **Yes** |
| ETHER-based baseline vs LLM-first pipeline (Sonnet 4.6) | 0.6115 | 0.6210 | -0.0095 | [-0.0816, +0.0548] | 0.8206 | No |
| AID-Dup vs LLM-first pipeline (Sonnet 4.6) | 0.7474 | 0.6210 | +0.1265 | [+0.0451, +0.2171] | 0.0002 | **Yes** |
| AID-Dup: Llama-3.1 vs Sonnet 4.6 (model-sensitivity range) | 0.6901 | 0.7474 | -0.0573 | [-0.1182, -0.0150] | < 0.0001 | **Yes** |
| LLM-first pipeline: Llama-3.1 vs Sonnet 4.6 (model-sensitivity range) | 0.1987 | 0.6210 | -0.4222 | [-0.5606, -0.2805] | < 0.0001 | **Yes** |
| AID-Dup vs LLM-first pipeline (Llama-3.1) | 0.6901 | 0.1987 | +0.4913 | [+0.3472, +0.6234] | < 0.0001 | **Yes** |
| AID-Dup vs LLM-first pipeline (Llama-4) | 0.7305 | 0.5072 | +0.2233 | [+0.1300, +0.3153] | < 0.0001 | **Yes** |
| AID-Dup vs LLM-first pipeline (Claude Haiku 4.5) | 0.7451 | 0.6122 | +0.1330 | [+0.0515, +0.2207] | 0.0004 | **Yes** |

## Reading the results

**Confirmed significant (CI excludes 0):**
- AID-Dup beats the ETHER-based baseline (delta -0.136, 12-series) -- the paper's central claim holds under series-resampling uncertainty, not just as a point estimate.
- AID-Dup beats the LLM-first pipeline at Sonnet 4.6 and at every other model tier tested (Llama-3.1, Llama-4, Haiku 4.5) -- the architecture comparison is robust regardless of which model backs the LLM-first pipeline.
- Both architectures show a real (non-zero) model-sensitivity range, but AID-Dup's range (Llama-3.1 to Sonnet, delta -0.057) is far narrower than LLM-first's (delta -0.422) -- consistent with the "robust to model choice" claim, though note the small Enhanced gap is itself statistically real, not literally zero.

**Not significant (CI includes 0) -- worth a precise sentence in the manuscript, not overclaiming:**
- ETHER-based baseline vs. LLM-first pipeline at Sonnet 4.6 (delta -0.0095, CI [-0.0816, +0.0548]). The point estimates (0.612 vs 0.621) suggest LLM-first is marginally ahead, but this is statistically indistinguishable from zero given only 12 series. The manuscript's existing framing ("barely ahead of the non-AI baseline") is actually MORE accurate stated this way -- consider revising to explicitly say the two are statistically indistinguishable, rather than implying a real (if small) LLM-first edge.
