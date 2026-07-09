"""
Deterministic FAERS deduplication pipeline -- a faithful re-implementation of:

  Kreimeyer K, Spiker J, Dang O, De S, Ball R, Botsis T.
  "Deduplicating the FDA adverse event reporting system with a novel
  application of network-based grouping." J Biomed Inform 165 (2025) 104824.
  (documents/dedup_paper.pdf + supplementary)

NO AI / LLM calls anywhere. The pipeline mirrors the paper's four components:

1. Pair Screening (paper 3.1.1, Fig 3): reports are assigned to parameter
   sets built from demographic, drug product, and adverse event data; a
   candidate pair must share a Product Active Ingredient (PAI) AND an
   Adverse Event (AE), with report dates within one year.
2. Pairwise Comparison (paper 3.1.2, Fig 1): a probabilistic record-linkage
   score over structured fields (age, sex, country, date, drug products)
   plus features extracted from the free-text narrative (diagnosis, medical
   history, family history, other symptoms, drug mentions -- the ETHER
   feature types), where matching information adds positive weight and
   mismatching information adds negative weight. High-scoring pairs go
   through a second component of regular-expression narrative searches
   (supplementary, Section B) before the final duplicate judgement.
3. Grouping and Group Splitting (paper 3.1.3): duplicate pairs form a
   network graph; connected components are recursively split with
   Clauset-Newman-Moore modularity optimization (NetworkX, per the
   supplementary) unless their edge density passes the paper's exemption
   thresholds: >0.7 (<50 reports), >0.8 (50-100), >0.9 (100-200),
   >0.95 (>200).
4. Reference Case Selection (paper 3.1.4): per group, the report with the
   largest sum of connecting edge weights; ties broken by the combined
   count of suspect products + adverse events, then by a completeness
   score, then by the most recent received date.

The exact per-field weights of the underlying probabilistic algorithm
(Kreimeyer et al. 2017, ref [21]) are not published in the available
documents; the defaults below follow the structure of Fig 1 (different
contribution per information type, positive for matches / negative for
mismatches) and are exposed in PAPER_CONFIG for calibration.
"""

import math
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations

try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
except ImportError:  # surfaced as a clean pipeline error by the caller
    nx = None

PAPER_CONFIG = {
    "screening": {
        # Fig 3c: date within one year of the original report.
        "date_window_days": 365,
    },
    "pairwise": {
        # match / mismatch weight per information type (log-odds style:
        # matches add, mismatches subtract, missing contributes nothing).
        "weights": {
            "sex":            {"match": 0.7,  "mismatch": -1.5},
            "age":            {"match": 1.5,  "mismatch": -2.0},
            "country":        {"match": 0.8,  "mismatch": -1.5},
            "date":           {"match": 1.5,  "mismatch": -1.0},
            "drugs":          {"match": 3.0,  "mismatch": -2.0},
            "events":         {"match": 2.5,  "mismatch": -2.0},
            # narrative (ETHER-type) features
            "diagnosis":      {"match": 2.0,  "mismatch": -1.0},
            "medical_history": {"match": 1.5, "mismatch": -0.5},
            "family_history": {"match": 1.0,  "mismatch": 0.0},
            "symptoms":       {"match": 1.0,  "mismatch": -0.5},
            "narrative_drugs": {"match": 1.5, "mismatch": -0.5},
            "narrative_dates": {"match": 2.0, "mismatch": 0.0},
        },
        "age_tolerance_years": 2,
        "date_exact_bonus_days": 0,     # full weight only on the same day...
        "date_partial_days": 30,        # ...half weight within a month
        # first-component threshold that sends a pair into the second
        # (regex) component, and the final duplicate thresholds.
        # calibrated on the 12 benchmark series (2026-07): thr=7.0 with
        # required second-component confirmation at 0.3 gave the best
        # precision-leaning F1, matching the paper's focus on minimizing
        # false discovery.
        "second_component_threshold": 4.5,
        "duplicate_threshold": 7.0,
        # a pair below duplicate_threshold can still be judged duplicate if
        # the second component confirms it and the first score is at least:
        "rescue_threshold": 5.5,
        "second_component_confirm": 0.3,
        # the paper: both scores "together rate ... whether they should be
        # judged to be duplicates" -- when True, a pair whose two reports
        # both have narratives is only a duplicate if the second (regex)
        # component also confirms it.
        "require_second_confirm": True,
    },
    "splitting": {
        # paper 3.1.3, hard-coded density exemptions by group size
        # (kept here only for visibility; get_required_density implements them)
    },
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "with", "was", "were", "is", "are",
    "to", "in", "on", "for", "at", "by", "from", "this", "that", "patient",
    "reported", "report", "unknown", "not", "no", "unk",
}


def _norm(value) -> str:
    return str(value).strip().lower() if value not in (None, "") else ""


def _to_set(val) -> set:
    if not val:
        return set()
    if isinstance(val, (list, set, tuple)):
        return {str(v).strip().lower() for v in val if str(v).strip()}
    if isinstance(val, str):
        return {val.strip().lower()}
    return set()


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _parse_age(value):
    try:
        age = float(value)
        return None if math.isnan(age) else age
    except (TypeError, ValueError):
        return None


def _phrase_tokens(phrases: set) -> set:
    """Content-word token set of a set of extracted phrases."""
    out = set()
    for p in phrases:
        out.update(t for t in _TOKEN_RE.findall(p) if len(t) >= 4 and t not in _STOPWORDS)
    return out


# --------------------------------------------------------------------------
# narrative feature extraction (deterministic ETHER-like patterns; the paper
# extracts diagnosis, medical history, family history, other symptoms, and
# drug products from the narrative)
# --------------------------------------------------------------------------

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

_PHRASE = r"([a-z][a-z0-9 ,'\-/]{2,60}?)(?=[.;:\n\)]|$| which | that | and was | and had )"
_FAMILY_HISTORY_RE = re.compile(r"family history(?: of|:)?\s+" + _PHRASE)
_MEDICAL_HISTORY_RE = re.compile(
    r"(?:medical history(?: of|:)?|history of|hx of|pmh(?: of|:)?)\s+" + _PHRASE)
_DIAGNOSIS_RE = re.compile(
    r"(?:diagnos(?:ed with|is of|is:|is was)|dx of)\s+" + _PHRASE)
_SYMPTOM_RE = re.compile(
    r"(?:experienced|developed|presented with|suffered(?: from)?|complained of)\s+" + _PHRASE)

_DATE_MMM_RE = re.compile(r"\b(\d{1,2})-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*-(\d{2,4})\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

# id-like alphanumeric tokens (lot numbers, control numbers, PMIDs...) used
# by the second (regex) component
_IDLIKE_RE = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z0-9][a-z0-9-]{5,}\b")


def extract_narrative_dates(text: str) -> set:
    """All parseable dates mentioned in a narrative, as ISO strings."""
    if not text:
        return set()
    text = str(text).lower()
    dates = set()

    for m in _DATE_MMM_RE.finditer(text):
        day, year = int(m.group(1)), int(m.group(3))
        if len(m.group(3)) == 2:
            year = 2000 + year if year < 50 else 1900 + year
        try:
            dates.add(datetime(year, _MONTHS[m.group(2)], day).strftime("%Y-%m-%d"))
        except ValueError:
            pass

    for m in _DATE_ISO_RE.finditer(text):
        try:
            dates.add(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d"))
        except ValueError:
            pass

    for m in _DATE_SLASH_RE.finditer(text):
        n1, n2, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if len(m.group(3)) == 2:
            year = 2000 + year if year < 50 else 1900 + year
        for month, day in ((n1, n2), (n2, n1)):
            try:
                dates.add(datetime(year, month, day).strftime("%Y-%m-%d"))
                break
            except ValueError:
                continue

    return dates


def extract_narrative_features(text: str, drug_vocabulary: set = None) -> dict:
    """One pass of deterministic pattern extraction per report. Returns the
    ETHER feature types the pairwise comparison uses (paper Fig 1)."""
    text_l = (str(text) if text else "").lower()
    family = {m.group(1).strip() for m in _FAMILY_HISTORY_RE.finditer(text_l)}
    # "history of" also matches inside "family history of" -- separate them.
    medical = {m.group(1).strip() for m in _MEDICAL_HISTORY_RE.finditer(text_l)} - family
    return {
        "diagnosis": _phrase_tokens({m.group(1).strip() for m in _DIAGNOSIS_RE.finditer(text_l)}),
        "medical_history": _phrase_tokens(medical),
        "family_history": _phrase_tokens(family),
        "symptoms": _phrase_tokens({m.group(1).strip() for m in _SYMPTOM_RE.finditer(text_l)}),
        "narrative_drugs": {w for w in (drug_vocabulary or set()) if w in text_l},
        "narrative_dates": extract_narrative_dates(text_l),
        "id_tokens": {t for t in _IDLIKE_RE.findall(text_l) if not _DATE_ISO_RE.fullmatch(t)},
        "has_text": bool(text_l.strip()),
    }


def build_drug_vocabulary(cases: list) -> set:
    """Distinct drug-name words (>=4 chars) across the series' structured
    drug fields; used to detect drug product mentions in narratives."""
    vocab = set()
    for c in cases:
        for d in _to_set(c.get("drugs")) | _to_set(c.get("drug")):
            vocab.update(w for w in _TOKEN_RE.findall(d) if len(w) >= 4)
    return vocab - _STOPWORDS


# --------------------------------------------------------------------------
# component 1: pair screening
# --------------------------------------------------------------------------

def screen_pairs(cases: list, config: dict = None) -> list:
    """Paper 3.1.1 / Fig 3: candidate pairs must share at least one PAI
    (drug product) AND one AE, with dates within one year. Implemented as an
    inverted index over (drug, event) parameter-set keys so the full n^2
    space is never enumerated."""
    cfg = (config or PAPER_CONFIG).get("screening") or PAPER_CONFIG["screening"]
    window_days = cfg.get("date_window_days", 365)

    drugs_by_case, events_by_case, date_by_case = {}, {}, {}
    for c in cases:
        cid = c["case_id"]
        drugs_by_case[cid] = _to_set(c.get("drugs")) | _to_set(c.get("drug"))
        events_by_case[cid] = _to_set(c.get("events")) | _to_set(c.get("event"))
        date_by_case[cid] = _parse_date(c.get("initial_date"))

    # parameter sets keyed on (PAI, AE)
    param_sets = defaultdict(list)
    for c in cases:
        cid = c["case_id"]
        for d in drugs_by_case[cid]:
            for e in events_by_case[cid]:
                param_sets[(d, e)].append(cid)

    candidates = set()
    for members in param_sets.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(set(members)), 2):
            candidates.add((a, b))

    screened = []
    for a, b in sorted(candidates):
        da, db = date_by_case[a], date_by_case[b]
        if da and db and abs((da - db).days) > window_days:
            continue
        screened.append((a, b))
    return screened


# --------------------------------------------------------------------------
# component 2: pairwise probabilistic comparison
# --------------------------------------------------------------------------

def _set_field_weight(set_a: set, set_b: set, w: dict) -> float:
    """Weight contribution of a set-valued field: overlap fraction scales the
    match weight; complete disjointness (both sides informative) counts as a
    mismatch; missing on either side contributes nothing."""
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b) / min(len(set_a), len(set_b))
    if overlap == 0.0:
        return w["mismatch"]
    return w["match"] * overlap


def compare_pair(a: dict, b: dict, feats_a: dict, feats_b: dict, config: dict = None) -> dict:
    """Paper 3.1.2 / Fig 1: probabilistic weight score over structured fields
    and narrative features, then (for high scorers) the second component of
    regex narrative searches. Returns {"weight_score", "second_score",
    "is_duplicate"}."""
    cfg = (config or PAPER_CONFIG).get("pairwise") or PAPER_CONFIG["pairwise"]
    w = cfg["weights"]
    score = 0.0

    # -- structured fields --------------------------------------------------
    sex_a, sex_b = _norm(a.get("sex")), _norm(b.get("sex"))
    if sex_a and sex_b:
        score += w["sex"]["match"] if sex_a == sex_b else w["sex"]["mismatch"]

    country_a, country_b = _norm(a.get("country")), _norm(b.get("country"))
    if country_a and country_b:
        score += w["country"]["match"] if country_a == country_b else w["country"]["mismatch"]

    age_a, age_b = _parse_age(a.get("age")), _parse_age(b.get("age"))
    if age_a is not None and age_b is not None:
        tol = cfg.get("age_tolerance_years", 2)
        score += w["age"]["match"] if abs(age_a - age_b) <= tol else w["age"]["mismatch"]

    da, db = _parse_date(a.get("initial_date")), _parse_date(b.get("initial_date"))
    if da and db:
        diff = abs((da - db).days)
        if diff <= cfg.get("date_exact_bonus_days", 0):
            score += w["date"]["match"]
        elif diff <= cfg.get("date_partial_days", 30):
            score += w["date"]["match"] * 0.5
        else:
            score += w["date"]["mismatch"]

    drugs_a = _to_set(a.get("drugs")) | _to_set(a.get("drug"))
    drugs_b = _to_set(b.get("drugs")) | _to_set(b.get("drug"))
    score += _set_field_weight(drugs_a, drugs_b, w["drugs"])

    events_a = _to_set(a.get("events")) | _to_set(a.get("event"))
    events_b = _to_set(b.get("events")) | _to_set(b.get("event"))
    score += _set_field_weight(events_a, events_b, w["events"])

    # -- narrative (ETHER-type) features -------------------------------------
    if feats_a["has_text"] and feats_b["has_text"]:
        for key in ("diagnosis", "medical_history", "family_history",
                    "symptoms", "narrative_drugs", "narrative_dates"):
            score += _set_field_weight(feats_a[key], feats_b[key], w[key])

    result = {"weight_score": round(score, 3), "second_score": None, "is_duplicate": False}

    if score >= cfg["duplicate_threshold"]:
        result["is_duplicate"] = True

    # -- second component: regex narrative searches (supplementary Sec. B) --
    both_texts = feats_a["has_text"] and feats_b["has_text"]
    if score >= cfg["second_component_threshold"] and both_texts:
        checks = []
        checks.append(1.0 if feats_a["narrative_dates"] & feats_b["narrative_dates"] else 0.0)
        checks.append(1.0 if feats_a["id_tokens"] & feats_b["id_tokens"] else 0.0)
        phrases_a = feats_a["diagnosis"] | feats_a["medical_history"] | feats_a["symptoms"]
        phrases_b = feats_b["diagnosis"] | feats_b["medical_history"] | feats_b["symptoms"]
        union = phrases_a | phrases_b
        checks.append(len(phrases_a & phrases_b) / len(union) if union else 0.0)
        second = sum(checks) / len(checks)
        result["second_score"] = round(second, 3)
        if (not result["is_duplicate"]
                and score >= cfg["rescue_threshold"]
                and second >= cfg["second_component_confirm"]):
            result["is_duplicate"] = True
        if (result["is_duplicate"]
                and cfg.get("require_second_confirm")
                and second < cfg["second_component_confirm"]):
            result["is_duplicate"] = False
    elif result["is_duplicate"] and cfg.get("require_second_confirm") and both_texts:
        # below the second-component gate: cannot be confirmed
        result["is_duplicate"] = False

    return result


# --------------------------------------------------------------------------
# component 3: grouping and group splitting
# --------------------------------------------------------------------------

def get_required_density(group_size: int) -> float:
    """Paper 3.1.3 exemption thresholds: density > 0.7 for groups smaller
    than 50 reports; > 0.8 for 50-100; > 0.9 for 100-200; > 0.95 above."""
    if group_size < 50:
        return 0.7
    if group_size <= 100:
        return 0.8
    if group_size <= 200:
        return 0.9
    return 0.95


def _split_recursive(graph) -> list:
    """Recursive Clauset-Newman-Moore modularity splitting; stops when a
    subgroup meets its density exemption or CNM makes no further split."""
    nodes = list(graph.nodes)
    if len(nodes) < 3:
        return [nodes]
    if nx.density(graph) > get_required_density(len(nodes)):
        return [nodes]
    communities = list(greedy_modularity_communities(graph, weight="weight"))
    if len(communities) <= 1:
        return [nodes]
    out = []
    for community in communities:
        out.extend(_split_recursive(graph.subgraph(community).copy()))
    return out


def group_and_split(duplicate_pairs: list) -> tuple:
    """duplicate_pairs: [(case_id_a, case_id_b, weight), ...] ->
    (final_groups, graph, n_initial_components). Groups of size 1 that fall
    out of splitting are dropped (a singleton is not a duplicate group)."""
    if nx is None:
        raise RuntimeError("The paper pipeline requires networkx (pip install networkx).")
    graph = nx.Graph()
    for a, b, weight in duplicate_pairs:
        graph.add_edge(a, b, weight=weight)

    components = list(nx.connected_components(graph))
    groups = []
    for component in components:
        subgraph = graph.subgraph(component).copy()
        groups.extend(_split_recursive(subgraph))
    groups = [sorted(g) for g in groups if len(g) >= 2]
    return groups, graph, len(components)


# --------------------------------------------------------------------------
# component 4: reference case selection
# --------------------------------------------------------------------------

def completeness_score(case: dict) -> float:
    """Fraction of populated key fields (stand-in for the completeness score
    of paper ref [27], which is not included in the available documents)."""
    fields = [
        _norm(case.get("age")), _norm(case.get("sex")), _norm(case.get("country")),
        _norm(case.get("initial_date")),
        "x" if (_to_set(case.get("drugs")) | _to_set(case.get("drug"))) else "",
        "x" if (_to_set(case.get("events")) | _to_set(case.get("event"))) else "",
        "x" if (case.get("narrative") or "").strip() else "",
    ]
    return sum(1 for f in fields if f) / len(fields)


def select_reference_case(group: list, graph, by_id: dict):
    """Paper 3.1.4: largest sum of connecting edge weights; then combined
    suspect products + AE count; then completeness; then most recent date."""
    def sort_key(cid):
        case = by_id[cid]
        weight_sum = sum(
            graph[cid][nbr].get("weight", 0.0)
            for nbr in graph.neighbors(cid) if nbr in group
        ) if graph.has_node(cid) else 0.0
        n_products_events = (
            len(_to_set(case.get("drugs")) | _to_set(case.get("drug")))
            + len(_to_set(case.get("events")) | _to_set(case.get("event")))
        )
        date = _parse_date(case.get("initial_date")) or datetime.min
        return (weight_sum, n_products_events, completeness_score(case), date)

    return max(group, key=sort_key)


# --------------------------------------------------------------------------
# full pipeline
# --------------------------------------------------------------------------

def run_dedup(cases: list, config: dict = None, progress=None) -> dict:
    """Runs all four components on a list of case dicts (the pipeline.py
    case shape). Returns groups with reference cases plus funnel counters.
    `progress(done, total)` is called during pairwise comparison."""
    config = config or PAPER_CONFIG
    by_id = {c["case_id"]: c for c in cases}

    # component 1
    candidate_pairs = screen_pairs(cases, config)

    # narrative features once per report involved in a candidate pair
    vocab = build_drug_vocabulary(cases)
    involved = {cid for pair in candidate_pairs for cid in pair}
    features = {
        cid: extract_narrative_features(by_id[cid].get("narrative"), vocab)
        for cid in involved
    }

    # component 2
    duplicate_pairs = []
    n_second = 0
    for i, (a, b) in enumerate(candidate_pairs):
        result = compare_pair(by_id[a], by_id[b], features[a], features[b], config)
        if result["second_score"] is not None:
            n_second += 1
        if result["is_duplicate"]:
            duplicate_pairs.append((a, b, result["weight_score"]))
        if progress:
            progress(i + 1, len(candidate_pairs))

    # component 3
    groups, graph, n_components = group_and_split(duplicate_pairs)

    # component 4
    groups_out = []
    for i, group in enumerate(groups, start=1):
        reference = select_reference_case(group, graph, by_id)
        groups_out.append({
            "group_id": i,
            "case_ids": [str(by_id[cid]["safety_report_id"]) for cid in group],
            "reference_case": str(by_id[reference]["safety_report_id"]),
        })

    return {
        "groups": groups_out,
        "funnel": {
            "reports": len(cases),
            "candidate_pairs_screened": len(candidate_pairs),
            "pairs_second_component": n_second,
            "duplicate_pairs": len(duplicate_pairs),
            "groups_before_split": n_components,
            "groups_after_split": len(groups_out),
        },
    }
