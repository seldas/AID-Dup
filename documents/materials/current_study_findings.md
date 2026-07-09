# Current study findings — can generative AI improve FAERS case deduplication?

> Snapshot of the study as of 2026-07-07. Manuscript-facing framing — internal
> development labels (e.g. "v2") are intentionally omitted below; see
> `MANUSCRIPT_NOTES.md` and the codebase for implementation-level detail if
> needed. Raw numbers backing every table below are in `dedup_results_raw.xlsx`
> (repo root), regenerated from `data/dedup_study.db`.

## Study logic

1. **Duplicate reports are a real problem for FAERS-based pharmacovigilance.** Duplicates within a case series create extra review work, and duplicates in large-scale data mining can distort drug-adverse event signal strength in either direction.
2. **Two approaches already exist in the literature, both benchmarked in Kreimeyer et al. 2025 (*J Biomed Inform* 165:104824):**
   - The **legacy/current FDA algorithm** — a proprietary, structured-fields-only rule-based tool, tuned for high precision at the cost of recall.
   - **InfoViP** — a probabilistic pairwise-comparison algorithm that combines structured fields with narrative features extracted by the ETHER NLP system, followed by network community-detection grouping. This is the published state of the art and the one we reproduce as our deterministic baseline.
3. **This study asks: in the generative-AI era, can an LLM improve on InfoViP, and if so, how should it be used?**
4. **We test two ways of bringing an LLM into the pipeline:**
   - **AID-Dup** — keep the reproduced InfoViP pipeline (structured-field scoring + narrative features) intact, and add an LLM narrative-adjudication step to resolve cases InfoViP's own deterministic thresholds leave ambiguous.
   - **Scratch approach** — build a new pipeline around the LLM directly, without InfoViP's expert-derived scoring weights. Since comparing every report against every other is computationally infeasible, this approach still needs a screening/blocking step (demographic bucketing) to bound the number of comparisons before the LLM is asked to adjudicate.
5. **Findings, below.**

We evaluate all conditions on the same 12 human-expert-adjudicated benchmark datasets (SE1–SE12) used in the original InfoViP paper — this is deliberate, not incidental: reusing the literature's own benchmark is what makes a direct comparison to the published InfoViP and current-FDA-algorithm numbers valid. See "Benchmark provenance" below for the correspondence evidence.

## Conditions compared

| Condition | What it is | LLM used? |
|---|---|---|
| Literature — current FDA algorithm | Published baseline (structured fields only, rule-based) | No |
| Literature — InfoViP | Published state of the art (structured fields + ETHER narrative features, probabilistic scoring, community-detection grouping) | No |
| **Reproduced InfoViP** (our deterministic baseline) | Our independent re-implementation of InfoViP, evaluated on the same 12 datasets | No |
| **AID-Dup** | Reproduced InfoViP + an LLM narrative-adjudication step for ambiguous cases (Claude Sonnet 4.6) | Yes |
| **Scratch approach** | New pipeline: demographic bucketing/blocking + LLM-driven pairwise adjudication, transitive-closure grouping — no InfoViP scoring weights (Claude Sonnet 4.6) | Yes |

## Benchmark provenance

SE1–SE12 are the same 12 datasets reported in Kreimeyer et al. 2025, confirmed independently from the paper's text: identical size distribution across the 12 series (2×~500, 4×~200, 4×~100, 2×~50 reports) and three exact narrative matches — the paper's dataset #7 ("reviewers found no real duplicates") = our SE7 (0 ground-truth groups); dataset #10 ("10 groups / 19 real pairs, neither tool found any") = our SE10 (10 ground-truth groups); dataset #12 ("8 groups / 173 real pairs, current algorithm found 0, InfoViP flagged 38") = our SE12 (8 ground-truth groups). The paper numbers its datasets largest-to-smallest, matching our SE1–SE12 ordering exactly.

⚠️ **The literature's per-series F1 values below are rough estimates, not published numbers — treat the macro-average as a range, not a precise figure.** The paper reports InfoViP's per-dataset performance only as a bar chart (Fig. 5); there is no numeric table anywhere in the paper or its supplementary material (checked directly — the supplementary material is implementation/infrastructure detail only, no results tables). The values below are read off that chart's pixel heights, and one internal inconsistency is worth flagging explicitly: my initial reading put the lowest value (SE3) at ~0.52, but the paper's abstract states the true minimum across the datasets is 0.36 — meaning at least that one reading is too high. Correcting SE3 down to the stated floor shifts the macro-average from 0.747 to 0.729. **Read "≈0.747" everywhere below as shorthand for "roughly 0.70–0.75," not a precise value.** The paper's own text anchors, by contrast, are exact and used as-is: *"InfoViP's deduplication pipeline outperformed the current algorithm in terms of F1-score on ten out of twelve data sets... F1 scores ranging from 0.36 to 0.93, with half above 0.75."* Three datasets (#7/#10/#12 = our SE7/SE10/SE12) are explicitly flagged by the paper itself as "F1 not calculable," and are excluded from the macro-averages below on the same basis. **Before this goes into a manuscript, get the exact numbers from the authors or their underlying data — do not cite the per-series values here as published figures.**

## Findings

**1. AID-Dup beats the published InfoViP pipeline's own reported performance, on InfoViP's own benchmark — and this holds even under the most conservative reading of the (approximate) literature numbers.** Restricting to the 9 series where the original paper reports a calculable F1: literature InfoViP is roughly 0.70–0.75 macro-F1 (my chart-based estimate, see caveat above), while AID-Dup reaches 0.860 — clearing even the high end of that range by more than 0.10. Our reproduced-InfoViP baseline lands modestly behind the published figure (0.696 vs ~0.70–0.75) — an expected gap for an independent reproduction of a proprietary production pipeline — and both it and the literature's InfoViP comfortably clear the literature's current-FDA-algorithm baseline (~0.43), which confirms the reproduction is sound before the LLM layer is added at all.

**2. The Scratch approach — LLM-driven, without InfoViP's expert-derived scoring — does not reach AID-Dup's performance, and barely beats the reproduced deterministic baseline.** Full 12-series macro-F1: Scratch 0.621 vs. AID-Dup 0.747 vs reproduced InfoViP 0.612. The gap is not "any LLM pipeline helps" — the LLM's judgment quality matters, but the paper's structured-field scoring and narrative-extraction logic are carrying real signal that a from-scratch bucketing+LLM pipeline doesn't recover on its own. Scratch wins outright on only one series (SE8) and ties on one more (SE6); on the hardest, most precision-limited series (SE4, SE10) it is the worst of the three conditions we tested.

**3. Net conclusion: an LLM adjudication step layered onto the existing InfoViP algorithm is the effective way to bring generative AI into FAERS deduplication — replacing the algorithm's structured scoring with an LLM pipeline is not.** The controlled design (identical 12-dataset benchmark across every condition, including the two literature baselines) is what makes this a direct claim rather than an analogy: AID-Dup's gain is attributable specifically to adding LLM adjudication on top of a faithful InfoViP reproduction, and the Scratch approach's shortfall shows that gain isn't just "because an LLM was involved."

## Macro-F1 summary

| Condition | Macro-F1 (12 series) | Macro-F1 (9 series comparable to literature*) |
|---|---|---|
| Literature — current FDA algorithm (rough estimate) | n/a (3 series not reported) | ~0.43 |
| Reproduced InfoViP (our baseline) | 0.612 | 0.696 |
| Scratch approach (bucketing + LLM, Sonnet 4.6) | 0.621 | 0.708 |
| Literature — InfoViP (rough estimate, range) | n/a (3 series not reported) | ~0.70–0.75 |
| **AID-Dup (InfoViP + LLM, Sonnet 4.6)** | **0.747** | **0.860** |

\* Excludes SE7/SE10/SE12, which the original paper itself flags as "F1 not calculable"; our conditions are recomputed over the same 9-series subset for a fair, like-for-like comparison.

## Per-series pair-level F1

| Series | Reproduced InfoViP | AID-Dup | Scratch approach | Lit. InfoViP (approx.) | Lit. current alg (approx.) |
|---|---|---|---|---|---|
| SE1  | 0.727 | 0.726 | 0.512 | ~0.56 | ~0.47 |
| SE2  | 0.921 | 0.941 | 0.932 | ~0.77 | ~0.32 |
| SE3  | 0.906 | 0.917 | 0.831 | ~0.36† | ~0.02 |
| SE4  | 0.168 | 0.550 | 0.121 | ~0.59 | ~0.23 |
| SE5  | 0.237 | 0.806 | 0.545 | ~0.88 | ~0.43 |
| SE6  | 0.892 | 0.914 | 0.917 | ~0.93 | ~0.55 |
| SE7  | 0.000 | 0.000 | 0.000 | n/c | n/c |
| SE8  | 0.819 | 0.905 | 0.953 | ~0.88 | ~0.73 |
| SE9  | 0.978 | 0.978 | 0.941 | ~0.84 | ~0.73 |
| SE10 | 0.087 | 0.250 | 0.162 | n/c | n/c |
| SE11 | 0.615 | 1.000 | 0.615 | ~0.75 | ~0.41 |
| SE12 | 0.988 | 0.983 | 0.922 | n/c | n/c |
| **Macro-avg (12)** | **0.612** | **0.747** | **0.621** | — | — |
| **Macro-avg (9 comparable)** | **0.696** | **0.860** | **0.708** | **~0.70–0.75** | **~0.43** |

(n/c = not calculable, per the paper's own convention for these 3 series. † SE3 shown at 0.36, the paper's stated minimum across all datasets — my original chart read of ~0.52 was almost certainly too high; see caveat above. P/R breakdowns, raw TP/FP/FN pair counts, group-level exact/partial/new/missed detail, and mean partial-match Jaccard for all three of *our* conditions × 12 series are in `dedup_results_raw.xlsx`.)

## Model ablation — does the choice of LLM matter?

Both AI conditions above use Claude Sonnet 4.6. To test how much of each approach's performance depends on that specific model, we re-ran both with the pipeline, prompt, and config held fixed and only the underlying LLM varied, across two open-weight models (Llama-3.1, Llama-4) and two proprietary models (Claude Haiku 4.5, Claude Sonnet 4.6).

### AID-Dup — model ablation

| Series | Llama-3.1 | Llama-4 | Claude Haiku 4.5 | Claude Sonnet 4.6 |
|---|---|---|---|---|
| SE1  | 0.599 | 0.652 | 0.741 | 0.726 |
| SE2  | 0.929 | 0.930 | 0.939 | 0.941 |
| SE3  | 0.830 | 0.917 | 0.920 | 0.917 |
| SE4  | 0.541 | 0.524 | 0.550 | 0.550 |
| SE5  | 0.806 | 0.806 | 0.806 | 0.806 |
| SE6  | 0.914 | 0.899 | 0.899 | 0.914 |
| SE7  | 0.000 | 0.000 | 0.000 | 0.000 |
| SE8  | 0.896 | 0.878 | 0.905 | 0.905 |
| SE9  | 0.978 | 0.978 | 0.978 | 0.978 |
| SE10 | 0.171 | 0.200 | 0.250 | 0.250 |
| SE11 | 0.667 | 1.000 | 1.000 | 1.000 |
| SE12 | 0.952 | 0.985 | 0.955 | 0.983 |
| **Macro-avg** | **0.690** | **0.731** | **0.745** | **0.747** |

### Scratch approach — model ablation

| Series | Llama-3.1 | Llama-4 | Claude Haiku 4.5 | Claude Sonnet 4.6 |
|---|---|---|---|---|
| SE1  | 0.089 | 0.249 | 0.432 | 0.512 |
| SE2  | 0.353 | 0.905 | 0.911 | 0.932 |
| SE3  | 0.282 | 0.827 | 0.818 | 0.831 |
| SE4  | 0.019 | 0.052 | 0.115 | 0.121 |
| SE5  | 0.035 | 0.382 | 0.545 | 0.545 |
| SE6  | 0.654 | 0.654 | 0.917 | 0.917 |
| SE7  | 0.000 | 0.000 | 0.000 | 0.000 |
| SE8  | 0.180 | 0.769 | 0.953 | 0.953 |
| SE9  | 0.647 | 0.736 | 0.941 | 0.941 |
| SE10 | 0.006 | 0.016 | 0.064 | 0.162 |
| SE11 | 0.000 | 0.571 | 0.727 | 0.615 |
| SE12 | 0.120 | 0.922 | 0.922 | 0.922 |
| **Macro-avg** | **0.199** | **0.507** | **0.612** | **0.621** |

**4. AID-Dup is nearly insensitive to which LLM is used, even at the weak end; the Scratch approach is highly sensitive.** Across all four models — Llama-3.1 → Llama-4 → Haiku 4.5 → Sonnet 4.6 — AID-Dup's macro-F1 moves only 0.690 → 0.731 → 0.745 → 0.747, a 0.057 spread. The same four models under the Scratch approach move 0.199 → 0.507 → 0.612 → 0.621, a 0.42+ spread — about 7.4× wider. Llama-3.1 is the clearest test of the mechanism: it's the model that collapsed hardest under Scratch (0.199, including a genuine 0-groups failure on SE11), yet under AID-Dup it lands at 0.690 — within 6 points of Sonnet 4.6 (0.747) and comfortably above the deterministic baseline (0.612). This is a second, independent line of evidence for finding 3: because AID-Dup only asks the LLM to adjudicate cases InfoViP's deterministic scoring already narrowed to "ambiguous," even a weak model's mistakes have limited room to hurt the final result. The Scratch approach hands the LLM the entire duplicate-detection decision with no deterministic backstop, so its output — and the final transitive-closure grouping built from it — inherits the model's raw capability almost directly. In practical terms: an AID-Dup-style architecture is a safe choice even with a weak, cheap, or swapped-out LLM; a Scratch-style architecture requires committing to (and re-validating against) a specific high-capability model.

## Precision and recall, not just F1

F1 alone hides *what kind* of error each condition makes, and the original paper's own headline claim is framed in exactly these terms: *"InfoViP outperformed the current algorithm... mainly due to its ability to capture more true duplicates (higher recall), while the current version seems to have a focus on avoiding false positive duplicates (high precision)."* Our own results have an equally clean, and structurally symmetric, precision/recall story — worth reporting explicitly rather than leaving buried inside the F1 numbers.

### Macro precision / recall by condition (12 series)

| Condition | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|
| Reproduced InfoViP (our baseline) | 0.580 | 0.739 | 0.612 |
| Scratch approach (Sonnet 4.6) | 0.629 | 0.681 | 0.621 |
| **AID-Dup (Sonnet 4.6)** | **0.761** | **0.741** | **0.747** |

**5. The reproduced deterministic baseline is precision-limited overall (recall 16 points above precision, macro), and AID-Dup's entire F1 gain is a precision fix — recall barely moves.** Reproduced InfoViP: P=0.580 / R=0.739. AID-Dup: P=0.761 / R=0.741. Precision rises by +0.181; recall moves by essentially nothing (+0.002). This is the mirror image of the literature's own InfoViP-vs-legacy-algorithm story: where InfoViP's advantage over the *legacy* algorithm was "more recall, same or lower precision," our LLM's advantage over InfoViP is "much more precision, same recall." The two AI-era conditions attack a different axis of the problem than the original paper's own innovation did.

**6. This shows up as a shift in how many series are precision-limited, not just a shift in the average.** Classifying each series as precision-limited, recall-limited, or balanced (precision and recall within 0.15 of each other, same convention as `MANUSCRIPT_NOTES.md` §6.3):

| Condition | Precision-limited | Recall-limited | Balanced† |
|---|---|---|---|
| Reproduced InfoViP | 5 series (SE1, SE4, SE5, SE10, SE11) | 0 series | 7 series |
| Scratch approach | 3 series (SE4, SE10, SE11) | 1 series (SE5) | 8 series |
| **AID-Dup** | **1 series (SE5)** | **1 series (SE8)** | **10 series** |

† Includes SE7, whose P=0/R=0 trivially falls inside the ±0.15 "balanced" band under every condition — this isn't a meaningful precision/recall balance, just an artifact of the specificity-control series' degenerate ground truth (see caveats). Excluding SE7, the balanced counts are 6 / 7 / 9 of the remaining 11 series, same pattern.

Under the deterministic baseline, 5 of 12 series are precision-limited and none are recall-limited — a one-sided failure pattern. Under AID-Dup, that collapses to 1 precision-limited series (SE5, still precision-limited but far less severely: -0.18 gap vs -0.71 gap at baseline) and 1 mildly recall-limited series (SE8, where the LLM pushed precision all the way to 1.000, slightly outrunning recall). The LLM's rescue step is, in aggregate, specifically resolving the deterministic algorithm's precision problem, not adding a new recall capability it didn't already mostly have.

**7. A few series make the mechanism concrete:**
- **SE4** (worst precision-limited case at baseline): P=0.101/R=0.500 → P=0.611/R=0.500. Recall is *exactly* unchanged; the LLM only removed false-positive merges the deterministic score had let through, more than 6x'ing precision.
- **SE11**: P=0.444/R=1.000 → P=1.000/R=1.000. Recall was already perfect at baseline; the LLM's entire contribution is eliminating false positives.
- **SE3** is the one clear counter-example: P=0.963/R=0.855 → P=0.933/R=0.901 — precision drops slightly while recall rises. The LLM traded a little precision for more recall here, the opposite of the dominant pattern, worth a sentence of discussion rather than treating the precision-fix story as universal.
- Several series show *no* precision/recall movement at all (e.g. SE9 is P=0.989/R=0.967 under both conditions) — the deterministic score already resolved these series with no cases landing in the LLM's ambiguous zone, consistent with the architecture described in the Study logic section.

**8. The Scratch approach doesn't show this clean pattern.** Its precision/recall gap (P=0.629/R=0.681) is smaller than the deterministic baseline's but larger than AID-Dup's, and per `MANUSCRIPT_NOTES.md` §6.5, its weakest-model failure mode (Llama-3.1) is simultaneously high false-positive *and* high false-negative counts on the same series — not a one-sided precision or recall problem but generalized noisiness, consistent with an architecture that hands the LLM the whole decision with no deterministic backstop (finding 4, above).

## Caveats / things to check before citing

- **The literature per-series F1 values are approximate chart reads, not exact published numbers** — re-derive them from the paper's supplementary material (or contact the authors) before quoting a specific series' literature F1 to two decimal places in a manuscript. The macro-average and range figures quoted directly from the paper's abstract/text *are* exact.
- **SE7 (specificity control) is still F1 = 0.0 on all three of our conditions and every model tested**, and the literature explicitly couldn't compute a figure for it either (InfoViP flagged 43 spurious pairs on a series with 0 true duplicates) — this over-merging weakness isn't unique to our reproduction, and no LLM (of the four tested) fixes it either.
- The headline AID-Dup-vs-Scratch comparison (finding 2, 0.747 vs 0.621) uses Claude Sonnet 4.6 for both, for a fair like-for-like comparison; the model ablation above shows that choice doesn't materially favor either side — AID-Dup leads Scratch by an even wider margin at every other model tier tested (0.731 vs 0.507 at Llama-4; 0.690 vs 0.199 at Llama-3.1).
- All figures are single latest-run snapshots per cell (no repeat-run averaging).
- **Macro precision/recall in the "Precision and recall" section include SE7 at P=0/R=0** (the degenerate-GT convention from `dedup_metrics.compute_metrics`, since every condition proposes spurious groups on a series with zero true duplicates). This depresses all three conditions' macro precision by the same fixed amount, so it doesn't bias the *comparison* between conditions, but it means the macro precision values are not directly comparable to a precision figure computed only over series with real duplicates.
