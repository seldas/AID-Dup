"""
Paper Approach v2 + AI (narrative-gated) -- an AI-ASSISTED variant of the
deterministic Paper Approach v2 pipeline (dedup_paper_v2.py).

v2's rule-based ETHER substitute (extract_narrative_features) is a documented
approximation: it is a stand-in for the FDA/JHU ETHER clinical NLP system,
which is not available here. This module does NOT replace that substitute
everywhere -- most reports already score correctly with it, and running an
LLM over every candidate pair's narratives would be prohibitively expensive
(a single series can screen tens of thousands of candidate pairs). Instead it
REPLACES the rule-based extraction only for reports in the small subset of
duplicate-candidate CLUSTERS where better narrative features are actually
likely to change the outcome. Every other component -- screening, scoring
weights/thresholds, grouping/splitting, reference-case selection -- is
identical to Paper Approach v2.

GATING IS AT THE CONNECTED-COMPONENT LEVEL, NOT THE REPORT LEVEL
------------------------------------------------------------------
An earlier per-report version of this gate (escalating only the specific
reports whose own rule-based extraction looked weak) was found, empirically,
to REDUCE recall on SE1-SE3 relative to pure v2 (2026-07 calibration run).
Root cause: the pairwise scoring is a token-overlap formula, and comparing an
AI-extracted report (precise, conservative token sets) against a
rule-based-extracted report (looser, more repetitive token sets) is an
apples-to-oranges comparison that the formula was never calibrated for --
even when the AI extraction is individually MORE accurate. Diagnostic
evidence: within one 13-report SE1 cluster, the 4 members that got AI
extraction stayed correctly grouped with EACH OTHER (AI-vs-AI comparisons
held up fine), while comparisons between those 4 and the other 9
(rule-based, untouched) members broke and fractured the cluster into two.

The fix: gate at the level of the rule-based duplicate-candidate CONNECTED
COMPONENT (the cluster of reports the deterministic pass would already group
together, before CNM splitting), not the individual report:

1. Score every candidate pair once with the normal config (structured fields
   + rule-based narrative features) to build the rule-based duplicate graph,
   and once with all narrative weights zeroed ("structured-only"). A pair is
   NARRATIVE-PIVOTAL if the two scores land on opposite sides of
   duplicate_threshold -- the narrative contribution is what decides its
   verdict, exactly where a better extraction could matter.
2. Take the connected components of the rule-based duplicate graph. Any
   component containing at least one narrative-pivotal pair is FLAGGED.
3. Every member of a flagged component -- not just the pivotal ones -- gets
   AI-assisted extraction. This guarantees every comparison inside that
   component is AI-vs-AI, never AI-vs-rule-based, so the token-overlap
   formula stays internally consistent. Components with no pivotal pair stay
   entirely rule-based (the common case), so cost still tracks only the
   clusters that actually need it -- typically a small fraction of a
   series's reports, not the whole series.

A MAX_COMPONENT_SIZE safety cap skips escalating a pathologically large
flagged component (e.g. a templated-batch cluster of hundreds of reports)
rather than letting cost balloon on an outlier case; this has not triggered
on the 12-series benchmark (largest pre-split component observed: 77).

THE GATE IS A FIXED POINT, NOT A SINGLE PASS
------------------------------------------------------------------
Step 2 above uses the RULE-BASED duplicate graph to decide the initial gate.
But AI-upgraded scores can create a duplicate edge the rule-based pass never
had -- including one that newly bridges a flagged component to an untouched
one. Left alone, that bridge would silently reintroduce an AI-vs-rule-based
comparison inside the FINAL group, the exact problem step 2 exists to
prevent (confirmed empirically: 17 such "mixed" final groups across the 12
series under the single-pass version, 2026-07). run_dedup_ai therefore
iterates: after each extraction round, it rebuilds the duplicate graph from
current scores and checks whether any resulting component still mixes gated
and non-gated members; if so, the non-gated side is pulled into the gate and
re-extracted, up to MAX_GATE_ROUNDS (converges within 1-2 extra rounds on
the 12-series benchmark).

Extraction results are cached persistently (db.narrative_extractions), keyed
by (narrative content hash, model option, prompt version), so different AI
models remain independently comparable and re-runs don't re-pay for
unchanged narratives.

NARRATIVE DOCUMENT-FREQUENCY FILTER -- NOT IN THE PAPER, UNIQUE TO THIS APPROACH
------------------------------------------------------------------
This module also applies filter_common_narrative_tokens (see
dedup_paper_v2.py) before gating and again before final scoring: it drops
any narrative token shared by more than NARRATIVE_FILTER_MAX_DOC_FREQ of the
series' reports, since such tokens are class-wide boilerplate (the series'
own suspect drug name, generic disease-history phrasing) rather than
case-specific evidence. This has no counterpart in the paper -- it was
introduced (2026-07) specifically to fix a precision collapse on SE5, and
the project owner decided it belongs only in this enhanced approach, not in
strict Paper Approach v2, since it is our own contribution rather than a
reproduction of anything published.
"""

import copy
import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import ai_client
import prompts
from db import connect, j, unj

import dedup_paper_v2

logger = logging.getLogger(__name__)

# Bumping this invalidates the persistent cache (a new key), so a prompt
# rewrite doesn't silently reuse extractions from the old prompt.
PROMPT_VERSION = 1

# Safety caps -- calibrated on the 12-series benchmark (2026-07); see module
# docstring. Exposed for the config API / transparency.
MAX_COMPONENT_SIZE = 150
# The gate is computed on the RULE-BASED duplicate graph, but AI-upgraded
# scores can create NEW duplicate edges the rule-based pass never had --
# including a bridge connecting a flagged component to an untouched one,
# which would silently reintroduce an AI-vs-rule-based comparison inside the
# FINAL group even though the initial gate was internally consistent. After
# each extraction round, run_dedup_ai rebuilds the duplicate graph and checks
# whether any resulting component still mixes gated and non-gated members;
# if so, the non-gated members are added to the gate and re-extracted, up to
# this many rounds (empirically converges within 1-2 extra rounds on the
# 12-series benchmark).
MAX_GATE_ROUNDS = 4

# NOT IN THE PAPER -- a document-frequency CUTOFF (not full TF-IDF weighting)
# unique to this enhanced approach, absent from strict Paper Approach v2.
# Series that concentrate on one drug (class) and one adverse event produce
# narrative vocabulary shared by nearly every report for reasons that have
# nothing to do with duplication (the series' own suspect drug name, generic
# disease-history phrasing); a token shared by more than this fraction of the
# series' involved reports is dropped from every report's feature set before
# scoring -- see dedup_paper_v2.filter_common_narrative_tokens. 0.20 is an
# EMPIRICAL calibration against our own 12-series benchmark (2026-07), not a
# literature value: swept 0.5 down to 0.15, 0.20 gives a large across-the-
# board F1 gain (mean 0.61 -> 0.73, confirmed cause of SE5's precision
# collapse: 0.14 -> 0.80) while staying clear of the point (~0.16-0.18) where
# it starts stripping genuine signal from the smallest series (SE12, 54 cases).
NARRATIVE_FILTER_MAX_DOC_FREQ = 0.20

GATE_CONFIG = {
    "max_component_size": MAX_COMPONENT_SIZE,
    "max_gate_rounds": MAX_GATE_ROUNDS,
    "narrative_filter_max_doc_freq": NARRATIVE_FILTER_MAX_DOC_FREQ,
    "prompt_version": PROMPT_VERSION,
}

_CATEGORIES = ("diagnosis", "medical_history", "family_history", "other_symptoms",
               "narrative_drugs", "narrative_dates")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------
# component-level gate: which reports need AI-assisted extraction
# --------------------------------------------------------------------------

def _structured_only_config(base_config: dict) -> dict:
    """Same config with all narrative-feature weights zeroed, so compare_pair
    scores as if the narrative didn't exist."""
    cfg2 = copy.deepcopy(base_config)
    for key in ("diagnosis", "medical_history", "family_history",
                "symptoms", "narrative_drugs", "narrative_dates"):
        cfg2["pairwise"]["weights"][key] = {"match": 0.0, "mismatch": 0.0}
    return cfg2


def _connected_components(duplicate_edges: list, node_universe: set) -> list:
    graph = dedup_paper_v2.nx.Graph()
    graph.add_nodes_from(node_universe)
    graph.add_edges_from(duplicate_edges)
    return list(dedup_paper_v2.nx.connected_components(graph))


def find_gated_components(byid: dict, candidate_pairs: list, config: dict = None) -> set:
    """Returns the set of case_ids that should get AI-assisted narrative
    extraction in place of the rule-based one (see module docstring): every
    member of any rule-based duplicate-graph connected component that
    contains at least one narrative-pivotal pair, so no comparison inside a
    flagged component ever mixes AI and rule-based features. This is the
    STARTING gate only -- see run_dedup_ai's convergence loop for why a
    single pass here is not sufficient by itself."""
    cfg = config or dedup_paper_v2.PAPER_V2_CONFIG
    structured_cfg = _structured_only_config(cfg)
    thr = cfg["pairwise"]["duplicate_threshold"]

    rb_duplicate_edges = []
    pivotal_pairs = []
    for a, b in candidate_pairs:
        va, vb = byid[a], byid[b]
        full = dedup_paper_v2.compare_pair(va, vb, cfg)
        if full["is_duplicate"]:
            rb_duplicate_edges.append((a, b))
        structured_score = dedup_paper_v2.compare_pair(va, vb, structured_cfg)["weight_score"]
        if (full["weight_score"] >= thr) != (structured_score >= thr):
            pivotal_pairs.append((a, b))

    involved = {cid for pair in candidate_pairs for cid in pair}
    components = _connected_components(rb_duplicate_edges, involved)
    component_of = {}
    for idx, comp in enumerate(components):
        for cid in comp:
            component_of[cid] = idx

    flagged_idx = set()
    for a, b in pivotal_pairs:
        idx_a, idx_b = component_of.get(a), component_of.get(b)
        if idx_a is not None:
            flagged_idx.add(idx_a)
        if idx_b is not None:
            flagged_idx.add(idx_b)

    gated = set()
    for idx in flagged_idx:
        component = components[idx]
        if len(component) > MAX_COMPONENT_SIZE:
            logger.warning(
                f"AI narrative gate: skipping a flagged component of "
                f"{len(component)} reports (> MAX_COMPONENT_SIZE={MAX_COMPONENT_SIZE})."
            )
            continue
        gated.update(component)
    return gated


# --------------------------------------------------------------------------
# AI-assisted extraction (cached, one call per unique narrative x model)
# --------------------------------------------------------------------------

def _narrative_hash(narrative: str) -> str:
    return hashlib.sha256((narrative or "").strip().encode("utf-8")).hexdigest()


def _cache_get(narrative_hash: str, model_option: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT features, input_tokens, output_tokens, cost FROM narrative_extractions "
            "WHERE narrative_hash = ? AND model_option = ? AND prompt_version = ?",
            (narrative_hash, model_option, PROMPT_VERSION),
        ).fetchone()
    if not row:
        return None
    return unj(row["features"], {}), {
        "input_tokens": row["input_tokens"] or 0,
        "output_tokens": row["output_tokens"] or 0,
        "cost": row["cost"] or 0.0,
    }


def _cache_put(narrative_hash: str, model_option: str, raw_features: dict, usage: dict):
    with connect() as conn:
        conn.execute(
            "INSERT INTO narrative_extractions(narrative_hash, model_option, prompt_version, "
            "features, input_tokens, output_tokens, cost) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(narrative_hash, model_option, prompt_version) DO NOTHING",
            (narrative_hash, model_option, PROMPT_VERSION, j(raw_features),
             usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("cost", 0.0)),
        )


def _clean_raw(raw) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    out = {}
    for key in _CATEGORIES:
        val = raw.get(key)
        out[key] = [str(x).strip() for x in val if str(x).strip()] if isinstance(val, list) else []
    return out


def _mock_raw_from_rule_based(rb_features: dict) -> dict:
    """Mock mode: fabricate the "AI" extraction from what the rule-based pass
    already found, so the pipeline is fully exercisable with no live endpoint."""
    return {
        "diagnosis": sorted(rb_features["diagnosis"]),
        "medical_history": sorted(rb_features["medical_history"]),
        "family_history": sorted(rb_features["family_history"]),
        "other_symptoms": sorted(rb_features["symptoms"]),
        "narrative_drugs": sorted(rb_features["narrative_drugs"]),
        "narrative_dates": sorted(rb_features["narrative_dates"]),
    }


def _phrase_tokens_ci(phrases) -> set:
    """Case-insensitive wrapper around dedup_paper_v2._phrase_tokens. The
    rule-based extractor always lowercases the full narrative before running
    its regexes, so its phrases are already lowercase; the AI naturally
    returns capitalized drug/diagnosis names (e.g. "Tocilizumab",
    "Prednisone"). _phrase_tokens' word regex only matches [a-z0-9]+, so an
    un-lowercased capitalized phrase gets silently mangled (leading capital
    dropped: "Tocilizumab" -> "ocilizumab"), which can never match the
    correctly-tokenized "tocilizumab" from another report -- corrupting
    overlap-based scoring for any capitalized AI output. Must lowercase here."""
    return dedup_paper_v2._phrase_tokens([str(p).lower() for p in phrases])


def _to_feature_sets(raw: dict, id_tokens: set) -> dict:
    """Converts the cached/fresh raw phrase-list dict into the token-set shape
    compare_pair expects (matching dedup_paper_v2.extract_narrative_features)."""
    dates = {d for d in raw.get("narrative_dates", []) if _DATE_RE.match(d)}
    return {
        "diagnosis": _phrase_tokens_ci(raw.get("diagnosis", [])),
        "medical_history": _phrase_tokens_ci(raw.get("medical_history", [])),
        "family_history": _phrase_tokens_ci(raw.get("family_history", [])),
        "symptoms": _phrase_tokens_ci(raw.get("other_symptoms", [])),
        "narrative_drugs": _phrase_tokens_ci(raw.get("narrative_drugs", [])),
        "narrative_dates": dates,
        "id_tokens": id_tokens,
        "has_text": True,
    }


def _extract_one(case: dict, view: "dedup_paper_v2.CaseView", model_option: str,
                  settings: dict, mock: bool, rule_based_features: dict) -> tuple:
    """Returns (feature_sets, usage, from_cache)."""
    narrative = case.get("narrative") or ""
    nhash = _narrative_hash(narrative)

    cached = _cache_get(nhash, model_option)
    if cached:
        raw, usage = cached
        return _to_feature_sets(raw, rule_based_features["id_tokens"]), usage, True

    if mock:
        raw = _mock_raw_from_rule_based(rule_based_features)
        usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    else:
        coded_events = ", ".join(sorted(view.pts)) or "(none coded)"
        user_msg = (
            prompts.DEDUP_NARRATIVE_EXTRACTION_USER_PROMPT
            .replace("{coded_events}", coded_events)
            .replace("{narrative}", narrative[:8000])
        )
        full_prompt = prompts.DEDUP_NARRATIVE_EXTRACTION_SYSTEM_INSTRUCTIONS + "\n\n" + user_msg
        text, usage = ai_client.generate(full_prompt, settings)
        raw = ai_client.parse_json_response(text)

    raw_clean = _clean_raw(raw)
    _cache_put(nhash, model_option, raw_clean, usage)
    return _to_feature_sets(raw_clean, rule_based_features["id_tokens"]), usage, False


# --------------------------------------------------------------------------
# full pipeline
# --------------------------------------------------------------------------

def run_dedup_ai(cases: list, config: dict = None, settings: dict = None,
                  mock: bool = False, progress=None, literature_aware: bool = False) -> dict:
    """Same four components as dedup_paper_v2.run_dedup; the only difference
    is that reports in gated connected components (see find_gated_components)
    get their narrative features from an LLM (cached) instead of the
    rule-based substitute.

    The gate is a fixed point, not a single pass: AI-upgraded scores can
    create a NEW duplicate edge the rule-based pass never had (including one
    that bridges a flagged component to an untouched one), which would
    silently reintroduce a mixed AI-vs-rule-based comparison inside the FINAL
    group. After each extraction round, the duplicate graph is rebuilt from
    current scores; if any resulting connected component still mixes gated
    and non-gated members, the non-gated members are pulled into the gate and
    re-extracted, up to MAX_GATE_ROUNDS. `progress(done, total)` is called
    across each round's newly gated reports (the AI step), not across all
    candidate pairs -- that count is what actually costs money/time here.

    `literature_aware`: when True (the "Enhanced + Literature-aware" variant,
    a separate approach in the app -- NOT the default Enhanced approach used
    for the main study results), a pair is forced to NOT be a duplicate,
    overriding the weight-score verdict, whenever both reports' narratives
    cite the SAME journal article (dedup_paper_v2.extract_literature_source).
    A published case-series article describes distinct patients by
    definition, so two reports citing it are a strong prior against
    duplication, not evidence for it -- see the SE7 root-cause analysis
    (documents/SE7_negative_control_analysis.md) for why this matters. This
    is a hard override applied only at final duplicate-pair assembly; it does
    not change screening, scoring, gating, or grouping otherwise, and
    defaults to False so the existing Enhanced-approach results are
    unaffected."""
    if dedup_paper_v2.nx is None:
        raise RuntimeError("The paper v2 pipeline requires networkx (pip install networkx).")
    config = config or dedup_paper_v2.PAPER_V2_CONFIG
    settings = settings or {}
    model_option = (settings.get("option_key") or "unknown") + (":mock" if mock else "")

    views = [dedup_paper_v2.CaseView(c) for c in cases]
    byid = {v.case_id: v for v in views}
    cases_by_id = {c["case_id"]: c for c in cases}

    candidate_pairs = dedup_paper_v2.screen_pairs(views, config)

    vocab = dedup_paper_v2.build_drug_vocabulary(views, cases)
    involved = {cid for pair in candidate_pairs for cid in pair}
    for cid in involved:
        v = byid[cid]
        raw = cases_by_id[cid].get("raw") or {}
        v.features = dedup_paper_v2.extract_narrative_features(
            v, cases_by_id[cid].get("narrative"), vocab,
            raw.get("Medical History/Medical History Comments"),
        )
    # strips class-wide boilerplate tokens (e.g. the series' own suspect drug
    # name, "diabetes mellitus" in a diabetes series) before the gate ever
    # looks at narrative overlap -- see dedup_paper_v2.filter_common_narrative_tokens.
    dedup_paper_v2.filter_common_narrative_tokens(byid, NARRATIVE_FILTER_MAX_DOC_FREQ)

    gated = find_gated_components(byid, candidate_pairs, config)

    usage_total = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    counts = {"cached": 0, "new": 0}
    lock = threading.Lock()
    cancelled = False

    def process_reports(cids):
        nonlocal cancelled
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(_process_one, cid): cid for cid in cids}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if progress:
                    try:
                        progress(completed, len(cids))
                    except Exception:
                        cancelled = True
                future.result()

    def _process_one(cid):
        nonlocal cancelled
        if cancelled:
            return
        v = byid[cid]
        rb_features = v.features
        try:
            feats, usage, from_cache = _extract_one(
                cases_by_id[cid], v, model_option, settings, mock, rb_features,
            )
        except Exception as e:
            logger.error(f"AI narrative extraction failed for case {cid}: {e}")
            feats, usage, from_cache = rb_features, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}, True
        with lock:
            if cancelled:
                return
            v.features = feats
            usage_total["input_tokens"] += usage.get("input_tokens", 0)
            usage_total["output_tokens"] += usage.get("output_tokens", 0)
            usage_total["cost"] += usage.get("cost", 0.0)
            counts["cached" if from_cache else "new"] += 1

    processed = set()
    rounds_used = 0
    for round_i in range(MAX_GATE_ROUNDS):
        rounds_used = round_i + 1
        to_process = gated - processed
        if to_process:
            process_reports(to_process)
            processed |= to_process
            if cancelled:
                raise RuntimeError("Cancelled")

        # Rebuild the duplicate graph with current (possibly mixed) features
        # and check whether any resulting component still mixes gated and
        # non-gated members -- if so, that's a new AI-created bridge; pull
        # the untouched side in and go another round.
        final_edges = [(a, b) for a, b in candidate_pairs
                       if dedup_paper_v2.compare_pair(byid[a], byid[b], config)["is_duplicate"]]
        components = _connected_components(final_edges, involved)
        newly_gated = set()
        for component in components:
            touched = component & gated
            if touched and len(touched) < len(component):
                newly_gated |= (component - gated)
        if not newly_gated:
            break
        gated |= newly_gated
    else:
        logger.warning(
            f"AI narrative gate: did not fully converge after {MAX_GATE_ROUNDS} rounds; "
            f"some cross-component AI/rule-based mixing may remain."
        )

    # re-apply the boilerplate filter now that AI extraction has replaced
    # some reports' features -- an AI extraction could itself produce
    # class-wide-common tokens, so the final scoring pass should see the same
    # document-frequency cleanup regardless of which reports were gated.
    dedup_paper_v2.filter_common_narrative_tokens(byid, NARRATIVE_FILTER_MAX_DOC_FREQ)

    literature_source = None
    lit_pairs_suppressed = 0
    if literature_aware:
        literature_source = {
            cid: dedup_paper_v2.extract_literature_source(cases_by_id[cid].get("narrative"))
            for cid in involved
        }

    duplicate_pairs = []
    for a, b in candidate_pairs:
        result = dedup_paper_v2.compare_pair(byid[a], byid[b], config)
        if not result["is_duplicate"]:
            continue
        if literature_source is not None:
            src_a, src_b = literature_source[a], literature_source[b]
            if src_a is not None and src_a == src_b:
                lit_pairs_suppressed += 1
                continue
        duplicate_pairs.append((a, b, result["weight_score"]))

    groups, graph, n_components = dedup_paper_v2.group_and_split(duplicate_pairs)

    groups_out = []
    for i, group in enumerate(groups, start=1):
        reference = dedup_paper_v2.select_reference_case(group, graph, byid)
        groups_out.append({
            "group_id": i,
            "case_ids": [str(byid[cid].safety_report_id) for cid in group],
            "reference_case": str(byid[reference].safety_report_id),
        })

    return {
        "groups": groups_out,
        "funnel": {
            "reports": len(cases),
            "candidate_pairs_screened": len(candidate_pairs),
            "gated_reports": len(gated),
            "gate_rounds": rounds_used,
            "ai_calls_new": counts["new"],
            "ai_calls_cached": counts["cached"],
            "duplicate_pairs": len(duplicate_pairs),
            "groups_before_split": n_components,
            "groups_after_split": len(groups_out),
            "literature_aware": literature_aware,
            "literature_pairs_suppressed": lit_pairs_suppressed,
        },
        "usage": usage_total,
    }
