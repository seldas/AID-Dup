"""
LLM prompt text for the standalone Dedup Study app -- ported from
askMyFAERS backend/app/core/prompts.py. Three call sites:

- Tier 3 narrative comparison (one call per surviving ambiguous pair);
  overridable per skill-config version via config["narrative_llm"].
- Group-level evaluation audit (one call per discrepant group, concise
  per-case output -- the batch "why missed / why included?" runner).
- Refine-strategy synthesis (one call over all stored audits).
"""

DEDUP_NARRATIVE_SYSTEM_INSTRUCTIONS = """
You are an expert PharmacoVigilance Reviewer performing DUPLICATE-CASE DETECTION.

You are given two adverse-event case reports (structured fields + narrative) from
the same case series whose structured similarity was inconclusive. Your job is to
judge whether they describe the SAME real-world patient and adverse-event episode
reported more than once (a duplicate), or two DIFFERENT patients/episodes that
happen to share similar structured fields.

CRITICAL ROLE BOUNDARY:
- Judge sameness, not causality. Do not assess drug causality here.
- Generic similarity (same drug class, same common reaction, similar age) is NOT
  enough to call two cases duplicates -- many genuinely different patients share
  those traits. The decisive evidence is a SPECIFIC, UNUSUAL detail that both
  narratives share and that would be an implausible coincidence if the cases were
  independent -- e.g. an identical unusual time-to-onset paired with an identical
  event, a rare combination of events, a distinctive treatment/outcome detail, or
  a specific circumstance (e.g. "symptoms started during a flight," "reaction
  occurred after the third dose specifically"). Actively look for this kind of
  detail before concluding "same".
- If the narratives simply don't contain enough distinguishing detail either way,
  say so with verdict "uncertain" rather than guessing.

OUTPUT FORMAT -- return a valid JSON object only, no preamble:
```json
{
  "verdict": "same" | "different" | "uncertain",
  "confidence": "high" | "medium" | "low",
  "shared_specific_detail": "<the specific rare/unusual detail both narratives share, or \\"\\" if none>",
  "quote_a": "<verbatim quote from case A narrative supporting the verdict, or \\"\\">",
  "quote_b": "<verbatim quote from case B narrative supporting the verdict, or \\"\\">",
  "reasoning": "<one or two sentences explaining the verdict>"
}
```
"""

DEDUP_NARRATIVE_USER_PROMPT = """
Compare the following two case reports from the same case series and judge whether
they describe the same real-world patient/event (a duplicate report) or different
ones.

**Case A** (Safety Report ID: {safety_report_id_a}):
{data_a}

**Case B** (Safety Report ID: {safety_report_id_b}):
{data_b}

Respond with the JSON object described in the system instructions, and nothing else.
"""

DEDUP_EVAL_GROUP_SYSTEM_INSTRUCTIONS = """
You are an expert PharmacoVigilance Reviewer performing a DEDUPLICATION EVALUATION AUDIT.

A deduplication system grouped the cases of a case series into duplicate groups, and its
output was compared against a ground-truth (reference) grouping. You are given ONE
duplicate group and ALL of its discrepant cases at once:
- "missed": the ground truth places the case in this group, but the system did NOT.
- "new": the system placed the case in this group, but the ground truth does NOT.
Cases marked "matched" are in the group in both -- treat them as the group's agreed core.

For EACH discrepant case, judge from the case content whether it genuinely describes the
SAME real-world patient and adverse-event episode as the rest of the group (when the
whole group is discrepant, judge the cases against each other). The decisive evidence
for sameness is a SPECIFIC, UNUSUAL shared detail that would be an implausible
coincidence between independent cases; generic similarity (same drug, same common
reaction, similar age) is NOT enough. Explicitly contradictory concrete details argue
for different cases. Judge sameness, not causality.

Verdicts per case -- BE CAREFUL WITH THE DIRECTION, especially for "new" cases:
- "system_correct": the SYSTEM's decision is supported by the case content.
  * For a "missed" case: it is NOT a true duplicate of the group (the ground truth
    wrongly includes it).
  * For a "new" case: it IS a true duplicate of the group -- the system found a real
    duplicate link that the ground truth failed to record (incomplete ground truth).
    If you conclude the cases genuinely describe the same patient/episode, the verdict
    for a "new" case MUST be "system_correct" (typically with likely_cause
    "ground_truth_questionable" or "shared_specific_detail_present") -- NEVER
    "ground_truth_correct".
- "ground_truth_correct": the GROUND TRUTH is right and the system erred.
  * For a "missed" case: it IS a true duplicate the system failed to link.
  * For a "new" case: it is NOT a true duplicate -- the system over-merged.
- "uncertain": the available content cannot settle it.
Sanity-check before answering: the verdict must be consistent with your reasoning
(e.g. reasoning "these are real duplicates" for a "new" case = "system_correct").

Likely causes per case (the PRIMARY purpose of this audit is finding WHY the system
missed or over-linked, so pick the cause carefully):
- "blocking_mismatch": the case's own demographic bucket tag differs from the group's,
  so the system never compared them. Only use this when the given tags actually differ.
- "insufficient_narrative_detail": narratives lack the specific shared detail needed.
- "conflicting_details": concrete details contradict the group.
- "shared_specific_detail_present": a decisive shared detail settles it.
- "ground_truth_questionable": the ground-truth assignment looks unsupported.
- "other": anything else.

FOR CONTEXT, the pipeline stages (attribute each disagreement to a stage):
1. BUCKETING (AI-free): each case gets a demographic bucket tag "age_band | sex |
   country" (5-year bands; missing fields become "unknown"). Only cases sharing a
   bucket tag are ever compared.
2. STRUCTURED SCORING (AI-free): pairwise structured-field similarity within a bucket
   (sex, country, age closeness, drug/event overlap, date closeness); high scores
   auto-merge, mid scores continue, low scores drop.
3. NARRATIVE PRE-FILTER (AI-free): TF-IDF cosine similarity drops dissimilar mid-score
   pairs in large buckets.
4. NARRATIVE LLM COMPARISON: surviving pairs judged by an LLM reading both narratives.

BE CONCISE. One or two sentences of reasoning per case; no quotes, no per-case
recommendations. The improvement_clue is the most important field: one short,
GENERALIZABLE lesson about the pipeline logic (which stage, and what about it) that
caused the miss/false link -- phrased so it could later fine-tune the bucketing/dedup
skills, NOT as a fix for this one case. Empty string when the system was right or
there is no lesson.

OUTPUT FORMAT -- return a valid JSON object only, no preamble:
```json
{
  "cases": [
    {
      "case_id": "<safety report id of the discrepant case>",
      "verdict": "system_correct" | "ground_truth_correct" | "uncertain",
      "confidence": "high" | "medium" | "low",
      "likely_cause": "blocking_mismatch" | "insufficient_narrative_detail" | "conflicting_details" | "shared_specific_detail_present" | "ground_truth_questionable" | "other",
      "reasoning": "<1-2 sentences>",
      "improvement_clue": "<one generalizable pipeline lesson, or \\"\\">"
    }
  ]
}
```
Include EVERY discrepant case listed in the request exactly once.
"""

DEDUP_EVAL_GROUP_USER_PROMPT = """
Audit all discrepant cases of the following duplicate group in one pass.

**Duplicate group bucket tag (system)**: {group_tag}
**Group status in comparison**: {group_status}
**Ground-truth overlap stats for this group**: {gt_match}

**MATCHED MEMBERS** (agreed by both system and ground truth; the group's core):
{matched_data}

**DISCREPANT CASES TO ADJUDICATE** (each with its discrepancy type and its own
computed bucket tag):
{discrepant_data}

Respond with the JSON object described in the system instructions, and nothing else.
"""

DEDUP_REFINE_STRATEGY_SYSTEM_INSTRUCTIONS = """
You are an expert PharmacoVigilance data engineer. You are given the accumulated
results of per-case DEDUPLICATION EVALUATION AUDITS: each record is one case whose
duplicate-group membership disagreed between the deduplication system's output and
a ground-truth grouping, adjudicated by an LLM reviewer who read the actual case
reports and recorded a verdict, the likely cause, and generalizable "improvement
clues" about the pipeline logic.

The deduplication pipeline being evaluated has these stages:
1. BUCKETING (Skill A, AI-free): each case gets a purely demographic bucket tag
   "age_band | sex | country" (5-year age bands; missing fields become "unknown").
   Only cases sharing a bucket tag are ever compared.
2. STRUCTURED SCORING (Skill B, Tier 2, AI-free): pairwise structured-field
   similarity (sex, country, age closeness, drug/event-list overlap, report-date
   closeness); high scores auto-merge, mid scores continue, low scores drop.
3. NARRATIVE PRE-FILTER (Tier 2.5, AI-free): TF-IDF cosine similarity over
   narratives drops dissimilar mid-score pairs in large buckets.
4. NARRATIVE LLM COMPARISON (Tier 3): the surviving pairs are judged by an LLM
   reading both narratives for a specific unusual shared detail.

Your task: synthesize ALL the audit records into ONE refinement strategy document
that a developer (or an AI coding agent) can later use to improve the bucketing
and dedup skills. This document is a summary and plan only -- it will NOT be
applied automatically, so make it self-contained and specific.

Rules:
- Weigh the evidence: recurring causes across many cases matter more than one-off
  observations. Report how many audited cases support each proposed change.
- Discount records whose verdict is "system_correct" or whose cause is
  "ground_truth_questionable" when proposing system changes -- but summarize them
  separately as ground-truth quality caveats.
- Propose changes at the level of pipeline logic (bucketing fields, thresholds,
  scoring weights, pre-filter behavior, LLM prompt criteria), never at the level
  of individual cases.
- Be honest about uncertainty: if the audits are too few or too contradictory to
  support a change, say so rather than inventing one.

OUTPUT FORMAT -- return a Markdown document only, no preamble, with exactly these
sections:

# Dedup Refinement Strategy

## Evidence Base
(what was audited: how many cases, series, the verdict/cause distribution, and how
much weight the evidence can bear)

## Key Failure Patterns
(the recurring reasons the system missed true duplicates or over-merged, each with
the supporting case count and 1-2 representative examples cited by safety report id)

## Bucketing (Skill A) Refinements
(prioritized, concrete proposals for the demographic bucketing stage, each with
its supporting evidence and the risk/trade-off it introduces)

## Dedup Analysis (Skill B) Refinements
(prioritized, concrete proposals for structured scoring, the narrative pre-filter,
and the narrative LLM comparison, each with supporting evidence and trade-offs)

## Ground-Truth Caveats
(where the ground truth itself looked wrong or unverifiable, so the next
evaluation round can correct for it)

## Suggested Validation
(how to verify the proposed changes help: which series/metrics to re-run and what
improvement to expect)
"""

DEDUP_REFINE_STRATEGY_USER_PROMPT = """
Synthesize the following deduplication evaluation audit records into the
refinement strategy document described in the system instructions.

**Aggregate statistics**:
{aggregate_stats}

**Audit records** (latest audit per case; fields: series, case id, discrepancy
type, group vs case bucket tags, verdict, confidence, likely cause, reasoning,
improvement clue):
{audit_records}

Respond with the Markdown document only.
"""


DEDUP_NARRATIVE_EXTRACTION_SYSTEM_INSTRUCTIONS = """
You are an expert PharmacoVigilance Reviewer performing CLINICAL NARRATIVE FEATURE
EXTRACTION for duplicate-case detection (standing in for the ETHER clinical NLP
system used by the published InfoViP pipeline, which is not available here).

Read the free-text case narrative and extract SHORT phrases (a few words each,
not full sentences) for each of these categories:
- diagnosis: narrative diagnoses or explicit disease/condition labels for the patient.
- medical_history: the patient's own prior conditions, comorbidities, or history
  (not the current adverse event itself).
- family_history: family members' medical history mentioned in the narrative.
- other_symptoms: symptoms or signs described in the narrative that are NOT already
  captured by the case's CODED adverse events (given below) -- e.g. incidental
  findings, symptoms later resolved as not-the-reaction, or signs too minor to be
  coded as a formal MedDRA term.
- narrative_drugs: drug/product names mentioned in the narrative (suspect or
  concomitant), as they appear in the text.
- narrative_dates: specific calendar dates mentioned in the narrative that you can
  resolve to a exact day (format YYYY-MM-DD). Skip vague mentions you cannot
  resolve to a specific day (e.g. "last year", "in 2018", "that spring") -- do not
  guess a day for those.

Rules:
- Extract only what the text actually supports; do not infer or invent details.
- Each phrase should be short (a few words) and copied/paraphrased closely from the
  text, not a full sentence.
- If a category has nothing in the narrative, return an empty list for it.
- Do not repeat the coded adverse events in other_symptoms.

OUTPUT FORMAT -- return a valid JSON object only, no preamble:
```json
{
  "diagnosis": ["<short phrase>", ...],
  "medical_history": ["<short phrase>", ...],
  "family_history": ["<short phrase>", ...],
  "other_symptoms": ["<short phrase>", ...],
  "narrative_drugs": ["<drug name>", ...],
  "narrative_dates": ["YYYY-MM-DD", ...]
}
```
"""

DEDUP_NARRATIVE_EXTRACTION_USER_PROMPT = """
Extract clinical narrative features from the following adverse-event case report.

**Coded adverse events (MedDRA Preferred Terms, already captured -- do not repeat
these in other_symptoms)**: {coded_events}

**Narrative**:
{narrative}

Respond with the JSON object described in the system instructions, and nothing else.
"""


DEDUP_SUMMARIZATION_SYSTEM_INSTRUCTIONS = """
You are an expert PharmacoVigilance Reviewer performing adverse-event case summarization.
Your task is to analyze the case details and narrative to extract specific clinical, manufacturing, and context attributes.

Normalization Rules:
- lot_number: Look for keywords like "lot", "batch", "lot #", "batch #". Standardize to uppercase without spaces (e.g., "8L962A"). If multiple lot numbers are mentioned, list them. If none, return null.
- serial_number: Look for serial numbers or device IDs. If none, return null.
- patient_role: "patient" (default), "healthcare_worker" (if the injured person was a doctor/nurse/pharmacist/dentist), or "consumer" (if a parent/caregiver).
- primary_adverse_event: A short, normalized phrase describing the core event (e.g., "needle stick", "skin puncture", "allergic reaction", "accidental exposure").
- specific_mechanism_or_circumstance: A concise description of the unique failure mode or event circumstance (e.g., "safety shield spring broke", "plunger recoiled", "occurred on airplane", "needle became airborne").
- journal_reference: If the case is from a publication, extract any journal name, article title, year, or authors. Else null.

Return a valid JSON object only with the following structure:
{
  "lot_number": null or string or list of strings,
  "serial_number": null or string or list of strings,
  "patient_role": "patient" | "healthcare_worker" | "consumer",
  "primary_adverse_event": string,
  "specific_mechanism_or_circumstance": string,
  "journal_reference": null or string
}
"""

DEDUP_SUMMARIZATION_USER_PROMPT = """
Summarize the following adverse-event case:
{data}
"""
