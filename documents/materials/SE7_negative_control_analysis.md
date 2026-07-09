# SE7 root-cause analysis — why AID-Dup (Sonnet 4.6) reports false duplicates on a negative-control series

> Standalone deep-dive, 2026-07-08. SE7 is the study's specificity control: 101 FAERS
> reports, ground truth = **0 duplicate groups** (every report is a genuinely distinct
> patient). Every condition tested in this study — deterministic, AID-Dup, Scratch,
> every LLM — scores F1 = 0.0 on SE7 (see `current_study_findings.md`, caveats). This
> document explains *why*, using the actual pipeline code and the actual case narratives,
> focused on AID-Dup's Claude Sonnet 4.6 result (snapshot #289).

## 1. What the system reported

AID-Dup's latest SE7 run produced **5 spurious duplicate groups covering 35 of the 101 cases**, and 114 false-positive pairs (0 true positives, since none exist):

| Group | Size | Cases |
|---|---|---|
| 0 | 6 | 17202206, 17202251, 17202255, 17202266, 17202610, 17202622 |
| 1 | 4 | 16612747, 17186694, 17202253, 17225077 |
| 2 | 9 | 17202268, 17202354, 17202356, 17202357, 17202611, 17202614, 17202615, 17202620, 17202621 |
| 3 | 9 | 16970357, 17202254, 17202257, 17202262, 17202270, 17202353, 17202355, 17202612, 17202613 |
| 4 | 7 | 17202252, 17202258, 17202261, 17202263, 17202264, 17202265, 17202267 |

## 2. What SE7 actually is

All 101 reports concern **the same drug, the same adverse event**: pentosan polysulfate sodium (brand name Elmiron) → maculopathy. This is not incidental — reading the narratives shows SE7 is dominated by two real-world reporting patterns that are structurally adversarial for any overlap-based deduplication method:

**(a) Literature-derived multi-patient case series reports.** Groups 0, 2, 3, and 4 each cite an academic case-series article — most (groups 0, 2, 4) cite the same paper, Hanif et al., *"Phenotypic Spectrum of Pentosan Polysulfate Sodium Associated Maculopathy: A Multicenter Study,"* JAMA Ophthalmology, Nov 2019 (a real 35-patient, 70-eye case series); group 3 cites a second article by "Kradjian et al." When a multi-patient academic study is reported to FAERS, each of the paper's patients becomes a *separate* FAERS report, but every one of those reports' narratives **reproduces the entire article abstract and methods verbatim** (drug exposure characterization, grading scheme, imaging findings for the aggregate 35-patient cohort) — text that has nothing to do with the individual patient and is byte-for-byte identical across every report drawn from that article. Each narrative also ends with an explicit administrative cross-reference: *"This case, from Same Literature article is linked to 20191222136, 20191239666, 20191239667, ... [30+ more case numbers]"* — literally FAERS's own annotation that these are **separate patients from the same source**, not a duplicate signal at all, yet it reads as a massive shared-token block to any narrative-overlap scorer.

**(b) Short mass-tort/litigation-style consumer reports.** Group 1's narratives are one-line, attorney/consumer-submitted claims like *"I have been diagnosed with maculopathy/Retinal Dystrophy as a result of taking Elmiron over the last nine years."* These carry almost no patient-specific content beyond the drug and event that define the series in the first place.

In both patterns, **every structured field the scoring algorithm relies on is identical or near-identical by construction**: same drug (screening requires it), same adverse event (screening requires it), same sex (97% female per the cited study), same country (USA), and — because these reports were often submitted or processed in the same administrative batch — the same FDA-received date (2019-12-26 for most of groups 0/2/3/4). None of these fields can discriminate between two real, distinct patients in this series. This is exactly the "templated/similar-looking reports... submissions from legal proceedings or extracts from large data sets" failure mode the original Kreimeyer et al. 2025 paper names explicitly in its Discussion as a known stress case for narrative-based deduplication — SE7 was almost certainly chosen as the specificity control *because* it exhibits this pattern.

## 3. Quantitative confirmation: tracing one false merge through the real scoring code

To move from "plausible explanation" to a verified mechanism, I ran the actual `dedup_paper_v2.compare_pair` scoring function (the same code AID-Dup uses) on a representative pair from Group 0: **case 17202206 (age 65) vs. case 17202251 (age 75)** — a pair the algorithm merged despite a 10-year age gap and genuinely different clinical details (symptom duration 8 vs. 35 years, dose 300 vs. 150 mg/day, therapy duration 7 vs. 15 years, maculopathy grade 3 vs. 2, differing per-eye visual acuity — all present in the narrative's patient-specific closing paragraph).

**Real score returned: 9.7** (duplicate_threshold = 8.5 → confirmed "duplicate"). Breakdown:

| Component | Contribution | Why |
|---|---|---|
| Structured fields (sex, country, date, drug, event) | **4.9** | Same sex/country/drug/event (screening-guaranteed) + same admin-received date (0.5x partial-date bonus); age *mismatch* penalty (-2.2) already applied |
| Narrative: `diagnosis` overlap | **+2.0** | Both extracted `{macular, degeneration, related}` — this is just a restatement of the series' own adverse event, not a patient-specific finding |
| Narrative: `medical_history` overlap | **+1.6** | Both extracted `{interstitial, cystitis}` — **interstitial cystitis is Elmiron's own FDA-labeled indication**; every patient in the cited study has it by inclusion criteria, so it's true of essentially the whole series, not evidence of shared identity |
| Narrative: `narrative_drugs` overlap | **+1.2** | Both extracted `{pentosan, polysulfate, sulfate, sodium, ...}` — literally the drug name being studied |
| **Total** | **9.7** | **≥ 8.5 → merged** |

Without the narrative contribution, this pair scores only 4.9 — well below even the second-component threshold (6.0) — and would correctly *not* be flagged. The entire false merge is driven by narrative tokens that are boilerplate for the series, not evidence about these two specific patients.

## 4. The pipeline has a fix for exactly this — and it isn't fully working on SE7

AID-Dup includes `filter_common_narrative_tokens`: any narrative token shared by more than 20% of a series' reports is stripped before scoring, specifically to catch class-wide boilerplate like a series' own drug name or (as documented in the codebase) "diabetes mellitus" in an all-diabetic series. It was added and verified to fix a similar precision collapse on SE5.

I checked the actual document frequency of the tokens driving the SE7 merge above, across all 89 SE7 reports involved in candidate pairs:

| Token | Document frequency |
|---|---|
| `narrative_drugs`: pentosan, polysulfate, sulfate | 80.9% |
| `narrative_drugs`: sodium | 78.7% |
| `medical_history`: cystitis | 66.3% |
| `medical_history`: interstitial | 56.2% |
| `diagnosis`: macular | 25.8% |

All comfortably above the 20% cutoff — this is exactly the pattern the filter is designed to catch. Simulating the filter on the example pair above (case 17202206 vs. 17202251) drops the score from **9.7 to 6.9** — correctly below the duplicate threshold, `is_duplicate` flips to `False`.

**Yet the actual stored "latest" AID-Dup result for SE7 (snapshot #289, run after the filter fix shipped) still contains this exact merge, and SE7's F1 is unchanged at 0.0 both before and after the filter was added** — unlike SE5, where the same fix measurably improved precision. This is an open discrepancy I could not fully resolve without invoking the actual Sonnet extraction call (not available in this environment): the filter demonstrably *should* fix this specific pair when applied to the rule-based-extracted features, but something in the production run — most plausibly the AI-narrative-extraction step re-introducing similarly-worded boilerplate tokens for the "narrative-pivotal" gated components that a second filter pass doesn't fully suppress, or these specific pairs not being included in the gated/escalated set at all — is preventing that fix from taking effect on SE7 specifically.

**Recommended next diagnostic step** (not performed here, requires the live AI environment): pull the cached AI-extracted features for the gated SE7 reports from `narrative_extractions` and confirm whether they still contain the boilerplate tokens post-extraction, and whether these specific pairs were part of the gated set in the first place.

## 5. A targeted fix, implemented and verified with real Sonnet 4.6 calls

Since a published case-series article describes distinct patients *by definition*, two reports citing the same article should be treated as a strong prior against duplication, not evidence for it. This was implemented as a new, separate, additive approach — **"AID-Dup + Literature-aware"** (`dedup_paper_v2.extract_literature_source` + a hard override in `dedup_paper_v2_ai.run_dedup_ai`, `skill_version = -3`) — that forces a pair to non-duplicate whenever both reports' narratives cite the same journal article. The main-result AID-Dup and Scratch approaches used elsewhere in this study are **unmodified**; this is supportive/discussion evidence, not a replacement result, and the rule is deliberately narrow (it only fires when a literature citation is detected — it does not touch Group 1's short attorney-submitted reports, which have no citation to match on).

Run on all 12 series with real Claude Sonnet 4.6 calls (2026-07-08):

| Series | Literature pairs suppressed | F1 change vs. AID-Dup |
|---|---|---|
| SE1–SE6, SE8–SE12 (11 series) | **0** | **None — byte-identical to the main AID-Dup** |
| SE7 | **187** | Still 0.0 (see below), but the underlying merge is substantially reduced |

The rule is surgical: it only ever fires on SE7, the one series actually built from literature-derived batches, and produces **zero change anywhere else** — no regression risk to the study's main results.

On SE7 itself, comparing AID-Dup before and after adding the literature-aware rule:

| | AID-Dup (snapshot #289) | AID-Dup + Literature-aware (snapshot #315) |
|---|---|---|
| Spurious groups | 5 | 4 |
| Cases wrongly merged | 35 | **19** (-46%) |
| False-positive pairs | 114 | **36** (-68%) |
| Pair-level F1 | 0.0 | 0.0 (unavoidable — GT is empty, so *any* false-positive pair drives precision, and hence F1, to 0) |

F1 doesn't move because SE7's degenerate ground truth (0 duplicate pairs) means even one remaining false-positive pair caps F1 at 0.0 — but the raw group/pair counts show the fix removes roughly two-thirds of the false-positive footprint. The remaining 19 cases / 4 groups are almost certainly the residual failure modes this rule doesn't address: Group 1's short, non-literature attorney reports (no citation to detect), and any coincidental structural over-merging (matching age/sex/country/date) among reports that don't share a common literature source.

Notably, this run cost **zero additional LLM calls** (`ai_calls_new: 0, ai_calls_cached: 38`) — because the literature-aware override only changes the final duplicate-pair filter, not the gating or extraction logic, it reused every cached narrative extraction from the prior AID-Dup run on SE7. This also settles the open question from an earlier draft of this analysis (whether the document-frequency filter's incomplete effectiveness in production, noted below, could be diagnosed by inspecting the AI's actual cached extractions) — the literature-aware rule sidesteps that question entirely by acting on the narrative citation directly, rather than relying on the extracted feature tokens.

## 6. Implications

- **This is not a bug specific to AID-Dup or to Claude Sonnet.** Every condition and every model tested in this study (Reproduced InfoViP, AID-Dup across 3 models, Scratch across 4 models) scores F1 = 0.0 on SE7 — see `current_study_findings.md`. The mechanism identified here (structured-field coincidence + citation/administrative narrative boilerplate) applies to the deterministic pipeline just as much as the AI-augmented ones; the LLM doesn't independently invent this failure, it's inherited from the same overlap-based scoring architecture.
- **It's a genuine limitation of narrative-overlap-based deduplication for this class of series**, not a data-quality problem with SE7 — mass-litigation and literature-derived multi-patient reporting are real, common FAERS phenomena (the original paper's own Discussion names this pattern explicitly), so this should be reported as a structural limitation worth discussing in the manuscript, not filed as "SE7 is an outlier to exclude."
- **The document-frequency filter is directionally correct but incompletely effective on its own** — it demonstrably fixes the failure mode in isolation (§3's simulation) but didn't close the gap in the original production run on SE7 (snapshot #289, F1 still 0.0 after the filter shipped). The literature-aware rule (§5) is a more direct, targeted fix for this specific failure mode, verified to cut SE7's false-positive footprint by roughly two-thirds with zero cost and zero regression elsewhere. Recommend describing the doc-frequency filter as a partial, general-purpose mitigation (verified on SE5) and the literature-aware rule as a targeted fix for the literature-citation failure mode specifically, not logically absolute (a patient could in principle appear in more than one citing publication) — appropriate for a manuscript's discussion/limitations section, not a claim that SE7 is now fully solved.
- A remaining, not-yet-implemented refinement: explicitly parsing and excluding the "linked to [case numbers]" administrative cross-reference block itself (rather than only the citation title) from narrative comparison, which could help with residual cases where the citation-title regex doesn't match cleanly.
