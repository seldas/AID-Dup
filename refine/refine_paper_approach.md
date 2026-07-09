Yes — focusing on the same 12 adjudicated series is sufficient. The bigger issue is that our current “Paper” comparator is still not close enough to the published InfoViP method. We should update the algorithm before finalizing the manuscript.

Differences to fix between our reimplementation and the literature

1. Candidate-pair screening is too simplified.
The published pipeline does not simply screen by shared drug + shared AE + date window. It first assigns each report to one or more parameter sets based on demographic, drug product, and adverse-event data, then searches “nearby” parameter sets sharing relevant similarities, then applies the final narrowing rule: shared Product Active Ingredient (PAI), shared AE, and date within one year. This workflow is shown in the page 4 Fig. 3 diagram of the published paper and described in the Pair Screening section.

Our current implementation approximates only the final narrowing step. We need to add a parameter-set layer that uses demographics plus PAI/AE/year-like attributes, then expands to nearby sets.

2. Product matching should be PAI-based, not raw drug-name overlap.
The paper explicitly requires Product Active Ingredient matching during screening. Our current notes say “shared drug product” and use drug/event inverted indexing. That is weaker and may mishandle brand/generic pairs such as Aimovig/erenumab or Byetta/exenatide.

We should normalize all product names to active ingredients or use an available RxNorm/FAERS product mapping before screening and scoring.

3. Narrative features should come from ETHER-style extraction, not generic regex.
The published algorithm uses structured fields plus ETHER-extracted narrative features. The feature categories explicitly named are diagnosis, medical history, family history, other symptoms, and drug products. The paper also says ETHER extracts clinical and temporal information from free-text narratives. PubMed’s abstract for the 2017 record-linkage paper likewise describes use of clinical and temporal information extracted by ETHER.

Our current implementation uses regex-derived diagnosis/history/symptom/drug/date-like features. That is conceptually similar but not implementation-consistent. We should regenerate these fields using an ETHER-like NLP module.

4. We need to reproduce the two-component pairwise comparison more faithfully.
The 2025 paper says the pairwise comparison outputs both a probabilistic weight score and a second component score, and that these together determine duplicate status. Our current Paper pipeline uses assumed thresholds: duplicate ≥7.0, rescue 5.5–7.0, second-component threshold 4.5, and regex rescue logic.

Those thresholds may be reasonable, but we should treat them as assumptions unless they came from the original 2017 method or supplement. At minimum, we need to document exactly which weights and thresholds are from the literature versus inferred.

5. Group splitting is mostly aligned, but should stay exactly as published.
Our density-gated CNM splitting matches the published description: recursive Clauset-Newman-Moore modularity optimization, with density exemptions of >0.7 for <50, >0.8 for 50–100, >0.9 for 100–200, and >0.95 for >200 reports. This part is one of the closest matches.

6. Reference-case selection is not needed for our 12-series pair/group benchmark.
The published full pipeline includes reference-case selection for data-mining use: choose the report with the largest sum of edge weights, then tie-break by suspect product + AE count, completeness score, and most recent received date. Since our endpoint is duplicate-pair/group recovery within the 12 case series, we do not need reference-case selection unless we later evaluate downstream deduplicated data-mining outputs.

ETHER features we need to regenerate or approximate

For the deduplication process, the published paper names these ETHER-derived narrative feature categories:

ETHER-derived feature	What to extract for our reimplementation
Diagnosis	Narrative diagnoses or explicit disease/condition labels
Medical history	Patient history, comorbidities, prior conditions
Family history	Family medical history mentions
Other symptoms	Symptoms/signs not already captured as coded FAERS AEs
Drug products	Drug names mentioned in the narrative, including concomitant/suspect products
Temporal information	Dates, durations, event timing, onset timing, sequence markers; this is described as part of ETHER’s broader clinical/temporal extraction, though the 2025 paper’s feature list emphasizes the five clinical categories above

The page 3 Fig. 1 example visually highlights narrative symptoms, products, and diagnosis as extracted evidence used in deduplication.

Practical update plan

I would revise the “Paper” algorithm in this order:

PAI normalization: map drug strings and narrative drug mentions to active ingredient concepts.
ETHER-style extraction: build a reproducible extractor for diagnosis, medical history, family history, symptoms, narrative drug products, and temporal expressions.
Parameter-set screening: assign each case to multiple parameter sets using PAI, AE, demographics, and date/year; expand to nearby sets before final PAI+AE+within-one-year filtering.
Pairwise score calibration: separate probabilistic score from second-component score and clearly document which weights/thresholds are literature-derived versus estimated.
Re-run the 12 series: compare new Paper-reimplemented results against the published Fig. 5 pattern and our current Paper baseline.
Then re-run AI-enhanced Paper: replace only the second-component ambiguous/rescue step with LLM adjudication, keeping the updated screening and scoring fixed.

