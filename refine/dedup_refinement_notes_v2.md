# Skill Refinement v1 → v2

Input: `dedup_audit_report_v1.md` (75 audits, latest per case, series SE10–SE12, all under skill v1).
Output: `dedup_skill_config_v2.json` — import it via **New skill version → 📂 Import JSON**, then
reset-and-rerun bucketing on each series and re-run the analysis to fill the v2 column.

## Evidence → change mapping

| # | Evidence (report section, count) | Change in v2 | Risk / trade-off |
|---|---|---|---|
| 1 | 52/75 audits carry a bucket mismatch; `blocking_mismatch` is the cause for 13. Unknown/`nan` **sex** blocked identical SE10 incidents (16731644, 16863887, 17439771/17564989…); **country** blocked all 7 SE12 literature re-reports of the same Kounis cases; unknown **age** blocked 15989585/16095902. | `bucketing.fields: ["age_band"]` — sex and country no longer hard-block; they still contribute to structured scoring. `age_band_years: 5 → 10` softens band-boundary splits. | Buckets get larger → more Tier-2 pairs and potentially more LLM calls; bounded by `max_group_size` 60 and the pre-filter. Unknown age still forms its own band (residual risk: 15989585-style misses). |
| 2 | SE10's 55-case "new" over-merge: identical drug+event with unknown demographics scores ~1.0 structurally, so distinct incidents auto-merged despite different **lot numbers** (13 `conflicting_details` audits), strengths, manufacturers, dates. | `high_confidence_threshold 0.85 → 0.92`; weights `drugs/events 2.0 → 1.5`, `date 1.0 → 1.5` — drug/event identity alone can no longer clear the auto-merge bar, and date disagreement pulls scores down harder. | True duplicates with sparse structured data now depend more on Tier 3; recall on structured-only duplicates may dip. |
| 3 | The over-merged cluster grew by union-find **chaining** (A~B, B~C ⇒ A,B,C). | `clustering_algorithm: "average_linkage"` (group-wide support required). | Slightly slower; may split genuinely chained follow-up trios. |
| 4 | 11 `insufficient_narrative_detail` over-merges passed Tier 3 at *medium* confidence on generic needle-stick similarity. | `merge_confidence: ["high"]` (was high+medium); pre-filter `similarity_threshold 0.15 → 0.2`. | Recall on weakly-documented true duplicates drops; those surface as ambiguous for human review instead. |
| 5 | Lot/batch, strength/dose, manufacturer, and incident-date conflicts were ignored by the narrative comparison; multi-patient reports were merged with single-patient ones; literature re-reports were the main GT-vs-system disagreement class. | Tier-3 prompt gains explicit rules: **hard conflicts** (stated-vs-stated lot/strength/manufacturer/date ⇒ "different"), a **common-event guard** (generic device injuries need a concrete shared identifier), a **multi-patient rule**, and a **literature re-report rule** (same published clinical signature ⇒ same case despite differing country/dates). | Longer prompt; slightly higher per-call tokens. |

## Not addressed by config (needs code or GT work)

- "Unknown value acts as wildcard" bucketing (case with unknown age matching any band) is not expressible in the config schema.
- Lot/batch number as a first-class structured scoring field (today it only exists inside narratives).
- The 25 `system_correct` audits point at **ground-truth gaps** (especially SE10 device-defect clusters and SE11 literature duplicates) — worth a GT curation pass before treating v2's precision/recall deltas as pure system effects.

## Validation plan

Re-run SE10–SE12 under v2 and compare the v1 vs v2 F1 columns. Expect: SE12 recall to rise sharply (7 country-blocked misses become comparable), SE10 precision to rise (lot/dose/date conflicts stop merging; no chaining), with the guard-rails possibly trading away a little SE10 recall. Then batch-audit the v2 runs to confirm the failure-cause distribution shifts away from `blocking_mismatch`.
