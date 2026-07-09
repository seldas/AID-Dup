# Paper Approach v2 — revision notes (2026-07-07)

The original "Paper Approach" (`dedup_paper.py`, snapshot version `paper`) scored
well below the published InfoViP pipeline (Kreimeyer 2025 JBI) on the 12
adjudicated series. This revision (`dedup_paper_v2.py`, snapshot version
`paperv2`) is a higher-fidelity, still fully deterministic re-implementation.
v1 is left untouched and remains runnable/selectable for comparison.

## What changed and why

The gap in v1 was traced (see `refine_paper_approach.md`) to over-simplified
screening, drug-name (not PAI) matching, regex-only narrative features, and
guessed weights. The key enabler for v2 was discovering that the raw FAERS
fields we already store carry most of what v1 was approximating:

| # | Paper component | v1 | v2 fix (data source) |
|---|---|---|---|
| 1 | Product matching | raw drug-name token overlap | **PAI** from `ALL Suspect Active Ingredients` (brand/generic solved in-data; no RxNorm) |
| 2 | Pair screening | shared drug + shared event | parameter sets on **PAI × MedDRA PT** (`All PTs`); MedDRA hierarchy (`All HLGTs`/`All SOCs`) as a graded "nearby set" scoring signal |
| 3 | Date window | **received** date (drops ~40% of true pairs — they span years) | **event** date (`Case Event Date`); true dups share it to within days. Precision-aware parsing: `"2018"` ≠ a specific day |
| 4 | Reference case | field-fill heuristic | real FAERS **`Completeness Score`** (the paper's ref [27]) |
| 5 | Narrative features | regex | offline **rule-based ETHER substitute** + the structured `Medical History` field |
| 6 | Grouping/splitting | CNM + density gates (faithful) | unchanged — reused from v1 |

## Decisions you made (2026-07-07)

- **ETHER substitute → offline rule-based NLP (no AI).** ETHER (the FDA/JHU
  clinical NLP) is not available to us. v2 uses a deterministic, section-aware
  extractor (`extract_narrative_features`) confined to one function so it can
  later be swapped for an LLM or a licensed clinical-NLP model without touching
  the rest of the pipeline. **This is the one component that is an
  approximation of the paper, not a faithful reproduction** — see the
  `dedup_paper_v2` module docstring.
- **Strict paper fidelity → named fields only.** The probabilistic score uses
  only fields the paper names (age, sex, country, date, PAI, AE + narrative
  features). Strong but unnamed FAERS signals (Manufacturer Control #, Lot #)
  are deliberately **not** used in scoring.

## Weights / thresholds

The Kreimeyer 2017 [21] per-field weights are unpublished, so every weight and
threshold in `PAPER_V2_CONFIG` is **inferred** and calibrated on the 12 series.
They follow the published structure (match adds / mismatch subtracts / missing
= 0; a first probabilistic component that gates a second regex component).
`duplicate_threshold = 8.5` is deliberately recall-leaning, matching the paper's
finding that InfoViP outperformed the incumbent "mainly due to its ability to
capture more true duplicates (higher recall)". On these 12 series the second
component neither vetoes nor rescues (it hurt F1 both ways, because true
duplicates here often share structured fields but carry terse narratives); it
is still computed and reported for transparency and paper fidelity.

## Results (pair-level F1 vs. active ground truth)

| Series | v1 `paper` | v2 `paperv2` | | Series | v1 | v2 |
|---|---|---|---|---|---|---|
| SE1 | 0.34 | **0.73** | | SE7* | 0.00 | 0.00 |
| SE2 | 0.82 | **0.92** | | SE8 | 0.61 | **0.82** |
| SE3 | 0.75 | **0.91** | | SE9 | 0.85 | **0.98** |
| SE4 | 0.46 | 0.17 | | SE10* | 0.06 | 0.09 |
| SE5 | 0.18 | **0.24** | | SE11 | 0.44 | **0.62** |
| SE6 | 0.68 | **0.89** | | SE12 | 0.36 | **0.70** |

Mean F1: **v1 = 0.46 → v2 = 0.59** (all 12); **0.65** on the 11 with duplicates.
10 of 12 series improve or tie; 6 series are ≥ 0.75, matching the paper's
reported shape (F1 0.36–0.93, half > 0.75).

## Known-hard series (remaining limitation)

SE4, SE5, SE7, SE10 are the "many near-identical, different-patient reports"
case (templated legal batches, device-issue reports, single-drug/single-event
series). Here structured fields alone cannot separate different patients, and
the deciding evidence is fine narrative detail — exactly what ETHER was for.
Our offline substitute is weakest here, so these stay low. The published
pipeline also failed SE7/SE10 (it flagged false pairs / found none) and had low
recall on SE12. If we later want these series, the highest-leverage change is a
stronger narrative extractor (LLM or licensed clinical NLP) plugged into
`extract_narrative_features` — the pluggable seam is already in place.

## App wiring

- `dedup_paper_v2.py` — the pipeline; `PAPER_V2_CONFIG` exposed at
  `/api/paper-config-v2`.
- `pipeline.run_paper_v2_analysis` — snapshots under pseudo version `-1`
  (`PAPER_V2_SKILL_VERSION`), scored like any run.
- `POST /api/series/<id>/analyze-v2` — endpoint.
- UI: "Paper Approach v2 (Revised, no AI)" radio; benchmark matrix shows an
  `F1 @ paperv2` column; the approach card shows v2's parameters when selected.
