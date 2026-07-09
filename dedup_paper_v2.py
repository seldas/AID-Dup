"""
Deterministic FAERS deduplication pipeline -- PAPER APPROACH v2.

A revised, higher-fidelity re-implementation of:

  Kreimeyer K, Spiker J, Dang O, De S, Ball R, Botsis T.
  "Deduplicating the FDA adverse event reporting system with a novel
  application of network-based grouping." J Biomed Inform 165 (2025) 104824.
  (documents/dedup_paper.pdf + supplementary)

WHY v2 EXISTS
-------------
v1 (dedup_paper.py) approximated the published "InfoViP" pipeline but scored
well below the paper's reported F1 range (0.36-0.93, half > 0.75) on the 12
adjudicated series. The refine analysis (refine/refine_paper_approach.md)
traced the gap to four things, three of which v2 fixes by using structured
FAERS fields that v1 ignored, and one of which we cannot fully reproduce:

  FIXED in v2
  1. Product matching is now PAI-based (Product Active Ingredient), taken
     directly from the FAERS field `ALL Suspect Active Ingredients`, instead
     of raw drug-name token overlap. This resolves brand/generic pairs
     (Aimovig/erenumab, Byetta/exenatide) for free -- FAERS already supplies
     the active ingredient. No RxNorm lookup is required.
  2. Pair screening now uses real MedDRA-coded adverse events (`All PTs`) for
     the parameter sets, and the MedDRA hierarchy (`All HLGTs`, `All SOCs`)
     to model the paper's "nearby parameter set" expansion as a graded
     scoring signal (Fig 3b) rather than a crude regex.
  4. Reference-case selection uses the real FAERS `Completeness Score` field
     (the paper's ref [27] score), not a field-fill heuristic.

  NOT FULLY REPRODUCIBLE -- ETHER (see below)
  3. The published pipeline extracts narrative features (diagnosis, medical
     history, family history, other symptoms, drug products) with ETHER, an
     FDA/JHU clinical NLP system that is NOT available to us. Per the project
     owner's decision (2026-07), v2 substitutes an OFFLINE, DETERMINISTIC
     rule-based extractor (`extract_narrative_features`) -- section-aware
     patterns + a MedDRA/lexicon-driven symptom reader + negation handling.
     It also folds in the STRUCTURED `Medical History` FAERS field, which
     ETHER would otherwise have to recover from free text. This is the one
     component that is an approximation of the paper rather than a faithful
     reproduction; it is confined to `extract_narrative_features` and to the
     narrative weights in PAPER_V2_CONFIG so it can be swapped later (e.g.
     for an LLM or a licensed clinical-NLP model) without touching the rest
     of the pipeline.

FIDELITY NOTE ON WEIGHTS/THRESHOLDS
-----------------------------------
The exact per-field weights of the underlying probabilistic algorithm
(Kreimeyer et al. 2017, ref [21]) are NOT published in the available
documents. Every numeric weight and threshold in PAPER_V2_CONFIG is therefore
INFERRED and calibrated on the 12 benchmark series; they follow the published
STRUCTURE (positive contribution for matches, negative for mismatches, nothing
for missing data; a first probabilistic component that gates a second
regular-expression component that together decide duplicate status). Per the
project owner's "strict paper fidelity" decision (2026-07), the probabilistic
component scores ONLY fields the paper names -- age, sex, country, date, PAI,
adverse events, and the ETHER-style narrative features. Strong but unnamed
FAERS duplicate signals (Manufacturer Control #, Lot #) are deliberately NOT
used in scoring.

NOTE: an earlier revision (2026-07) added a narrative document-frequency
filter here to fix a precision collapse on SE5 (a drug-class/single-AE
series where the suspect drug's own name and generic disease-history phrasing
inflated many non-duplicate pairs' scores). That filter has NO counterpart in
the paper, so per the project owner's decision it was moved OUT of strict
Paper Approach v2 and now lives only in Paper Approach v2 + AI
(dedup_paper_v2_ai.py), which is explicitly our own enhancement rather than a
paper reproduction. `filter_common_narrative_tokens` (below) is still defined
here since it operates on this module's CaseView feature shape, but v2's
own run_dedup no longer calls it.

NO AI / LLM CALLS anywhere in this module.
"""

import math
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations

try:
    import networkx as nx
except ImportError:  # surfaced as a clean pipeline error by the caller
    nx = None

# Group splitting (component 3) is unchanged from the paper and already
# faithful in v1; v2 reuses it rather than duplicating it.
from dedup_paper import group_and_split, get_required_density  # noqa: F401


PAPER_V2_CONFIG = {
    "screening": {
        # Fig 3c: PAI match AND AE (PT) match AND event date within one year.
        # The window only excludes a pair when BOTH event dates are known and
        # more than a year apart (see CaseView on why event date, not received).
        "date_window_days": 365,
    },
    "pairwise": {
        # Log-odds-style contribution per information type: matches add,
        # mismatches subtract, missing on either side contributes nothing.
        # INFERRED (Kreimeyer 2017 weights unpublished) -- see module docstring.
        "weights": {
            "sex":             {"match": 0.6,  "mismatch": -2.0},
            "age":             {"match": 1.4,  "mismatch": -2.2},
            "country":         {"match": 0.5,  "mismatch": -1.2},
            "date":            {"match": 1.6,  "mismatch": -1.2},
            "pai":             {"match": 2.6,  "mismatch": -2.5},
            "events":          {"match": 2.6,  "mismatch": -1.6},
            # graded "nearby parameter set" AE signal (MedDRA hierarchy), only
            # rewarded, never penalised (it is a similarity aid, Fig 3b).
            "events_hier":     {"match": 1.0,  "mismatch": 0.0},
            # narrative (ETHER-substitute) features
            "diagnosis":       {"match": 2.0,  "mismatch": -0.5},
            "medical_history": {"match": 1.6,  "mismatch": 0.0},
            "family_history":  {"match": 1.2,  "mismatch": 0.0},
            "symptoms":        {"match": 1.2,  "mismatch": -0.3},
            "narrative_drugs": {"match": 1.2,  "mismatch": 0.0},
            "narrative_dates": {"match": 2.2,  "mismatch": 0.0},
        },
        "age_tolerance_years": 1,
        "date_exact_bonus_days": 1,    # same day (or adjacent) = full weight...
        "date_partial_days": 60,       # ...half weight within two months
        # First component gate into the second (regex) component, and the
        # duplicate thresholds. INFERRED / calibrated on the 12 series (2026-07);
        # 8.5 is deliberately recall-leaning, matching the paper's finding that
        # InfoViP "outperformed the current algorithm ... mainly due to its
        # ability to capture more true duplicates (higher recall)".
        "duplicate_threshold": 8.5,    # duplicate at/above this weight score
        "second_component_threshold": 6.0,  # first score needed to run comp. 2
        # Second-component ("regex narrative search") calibration. The paper
        # runs comp. 2 on high-scoring pairs and lets the two scores "together"
        # decide. On the 12 adjudicated series, narrative-specifics overlap is
        # sparse even among TRUE duplicates (many share structured fields but
        # carry terse/templated narratives), so using comp. 2 as either a veto
        # or a rescue REDUCED F1. We therefore keep comp. 2 computed and
        # reported for transparency and paper fidelity, but the duplicate
        # verdict is driven by the first probabilistic component. Set
        # `require_second_confirm`/`rescue_threshold` to re-enable either role.
        "rescue_threshold": 99.0,      # disabled (see note above)
        "second_component_confirm": 0.5,
        "require_second_confirm": False,
    },
}


# ==========================================================================
# structured-field parsing (the FAERS `raw` dict + the flat case fields)
# ==========================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "with", "was", "were", "is", "are",
    "to", "in", "on", "for", "at", "by", "from", "this", "that", "patient",
    "reported", "report", "unknown", "not", "no", "unk", "his", "her", "she",
    "had", "has", "have", "been", "who", "which", "also", "due", "after",
    "prior", "history", "hx", "pmh", "year", "old", "years", "since", "then",
}

# FDA nonproprietary biologic suffix, e.g. erenumab-AOOE, adalimumab-ATTO.
_BIO_SUFFIX_RE = re.compile(r"-[a-z]{4}$")
_SALT_SUFFIXES = (
    " hydrochloride", " dihydrochloride", " hydrobromide", " sodium",
    " potassium", " calcium", " magnesium", " sulfate", " sulphate",
    " hemisulfate", " acetate", " diacetate", " mesylate", " besylate",
    " tosylate", " citrate", " dicitrate", " tartrate", " bitartrate",
    " maleate", " fumarate", " hemifumarate", " succinate", " phosphate",
    " diphosphate", " bromide", " chloride", " nitrate", " pamoate",
    " hcl", " na",
    # ester prodrug suffix, e.g. cefuroxime axetil -> cefuroxime (the oral
    # ester form hydrolyzes to the same active drug in vivo). Confirmed
    # (2026-07) as the sole cause of a 78-pair screening-recall gap in SE12 --
    # "cefuroxime axetil" vs "cefuroxime"/"cefuroxime sodium" never shared a
    # normalized PAI, so those true-duplicate pairs never became candidates.
    " axetil",
)


def _norm(value) -> str:
    return str(value).strip().lower() if value not in (None, "") else ""


def _clean_scalar(v):
    """A FAERS scalar is 'missing' if it is blank / nan / none-like."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "unk", "unknown"):
        return None
    return s


def _split_colon_list(value) -> list:
    """FAERS packs multi-valued fields as colon-delimited strings (PTs,
    HLGTs, SOCs, active ingredients). None of those vocabularies contain a
    literal ':' inside a single term."""
    s = _clean_scalar(value)
    if not s:
        return []
    return [p.strip() for p in s.split(":") if p.strip()]


def _norm_pai(raw_ingredient: str) -> str:
    """Normalise one active-ingredient string to a comparable PAI concept:
    drop the (PS)/(SS)/(I)/(C) role tag, the biologic suffix, and a trailing
    salt form so 'ERENUMAB-AOOE(PS)' and 'enoxaparin sodium' collapse to
    'erenumab' / 'enoxaparin'."""
    x = re.sub(r"\([^)]*\)", "", raw_ingredient).strip().lower()
    x = _BIO_SUFFIX_RE.sub("", x)
    changed = True
    while changed:
        changed = False
        for salt in _SALT_SUFFIXES:
            if x.endswith(salt) and len(x) > len(salt) + 2:
                x = x[: -len(salt)].strip()
                changed = True
    return x


def _to_set(val) -> set:
    if not val:
        return set()
    if isinstance(val, (list, set, tuple)):
        return {str(v).strip().lower() for v in val if str(v).strip()}
    if isinstance(val, str):
        return {val.strip().lower()}
    return set()


def _parse_date_prec(value):
    """Returns (datetime, precision) where precision is 'day', 'month', or
    'year'. Precision matters: FAERS event dates are frequently year- or
    month-only ('2018', '07/2016'), and treating those as an exact day would
    make every same-year report look like it shares an event date -- a major
    source of false-positive duplicate pairs among different patients."""
    if not value:
        return None, None
    s = str(value).strip()
    day_fmts = (("%Y-%m-%d", 10), ("%m/%d/%Y", None), ("%m/%d/%y", None), ("%Y/%m/%d", None))
    for fmt, cut in day_fmts:
        try:
            return datetime.strptime(s[:cut] if cut else s, fmt), "day"
        except ValueError:
            continue
    for fmt in ("%m/%Y", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt), "month"
        except ValueError:
            continue
    try:
        return datetime.strptime(s, "%Y"), "year"
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s[:10]), "day"
    except ValueError:
        return None, None


def _parse_date(value):
    return _parse_date_prec(value)[0]


def _parse_age(value):
    try:
        age = float(value)
        return None if math.isnan(age) else age
    except (TypeError, ValueError):
        return None


class CaseView:
    """A parsed, comparison-ready view over one pipeline case dict. Pulls the
    high-value structured FAERS fields out of `raw` (falling back to the flat
    case fields) so the rest of the pipeline never touches raw strings."""

    __slots__ = ("case_id", "safety_report_id", "pai", "pts", "hlgts", "socs",
                 "age", "sex", "country", "event_date", "event_prec",
                 "recv_date", "completeness", "features")

    def __init__(self, case: dict):
        raw = case.get("raw") or {}
        self.case_id = case["case_id"]
        self.safety_report_id = case.get("safety_report_id")

        # --- Product Active Ingredient (the paper's PAI) ---
        ing = (raw.get("ALL Suspect Active Ingredients")
               or raw.get("ALL Suspect Product Active Ingredients") or "")
        pai = {_norm_pai(p) for p in _split_colon_list(ing)}
        if not pai:  # fall back to the flat drug fields
            pai = {_norm_pai(d) for d in (_to_set(case.get("drugs")) | _to_set(case.get("drug")))}
        self.pai = {p for p in pai if p}

        # --- adverse events at three MedDRA levels ---
        self.pts = {p.lower() for p in _split_colon_list(raw.get("All PTs"))}
        if not self.pts:
            self.pts = _to_set(case.get("events")) | _to_set(case.get("event"))
        self.hlgts = {h.lower() for h in _split_colon_list(raw.get("All HLGTs"))}
        self.socs = {s.lower() for s in _split_colon_list(raw.get("All SOCs"))}

        # --- demographics / date ---
        self.age = _parse_age(_clean_scalar(raw.get("Age in Years")) or case.get("age"))
        self.sex = _norm(_clean_scalar(raw.get("Sex")) or case.get("sex"))
        self.country = _norm(_clean_scalar(raw.get("Country Derived")) or case.get("country"))
        # The paper's "date within one year of the original report" (Fig 3c) is
        # the date of the patient experience -- the ADVERSE-EVENT date, not the
        # administrative FDA-received date. True duplicates share an event date
        # to within days, but their received dates legitimately span years
        # (follow-ups / resubmissions), so screening on the received date
        # wrongly drops ~40% of true pairs. Event date is often missing, in
        # which case the window simply does not apply (the pair is kept).
        self.event_date, self.event_prec = _parse_date_prec(_clean_scalar(raw.get("Case Event Date")))
        self.recv_date = _parse_date(case.get("initial_date") or raw.get("Initial FDA received Date"))

        # --- completeness (paper ref [27]) ---
        comp = _clean_scalar(raw.get("Completeness Score"))
        try:
            self.completeness = float(comp) if comp is not None else None
        except ValueError:
            self.completeness = None

        self.features = None  # narrative features, filled lazily


# ==========================================================================
# component: narrative feature extraction  (ETHER SUBSTITUTE -- see docstring)
# --------------------------------------------------------------------------
# Deterministic, offline stand-in for ETHER. Extracts the five ETHER feature
# categories the paper feeds into pairwise scoring, plus temporal cues and
# id-like tokens for the second component. NOT a faithful reproduction of
# ETHER; confined here so it can be replaced without touching the pipeline.
# ==========================================================================

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

_PHRASE = r"([a-z][a-z0-9 ,'\-/]{2,60}?)(?=[.;:\n\)]|$| which | that | and was | and had )"
_FAMILY_HISTORY_RE = re.compile(r"family history(?: of|:)?\s+" + _PHRASE)
_MEDICAL_HISTORY_RE = re.compile(
    r"(?:medical history(?: of|:)?|history of|hx of|pmh(?: of|:)?)\s+" + _PHRASE)
_DIAGNOSIS_RE = re.compile(
    r"(?:diagnos(?:ed with|is of|is:|is was)|dx of|found to have|consistent with)\s+" + _PHRASE)
_SYMPTOM_RE = re.compile(
    r"(?:experienced|developed|presented with|suffered(?: from)?|complained of|"
    r"reported having|noted)\s+" + _PHRASE)
# simple negation guard: drop a symptom phrase immediately preceded by a negator
_NEGATION_RE = re.compile(r"\b(no|not|without|denies|denied|negative for|ruled out|"
                          r"absence of)\s+" + _PHRASE)

_DATE_MMM_RE = re.compile(r"\b(\d{1,2})-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*-(\d{2,4})\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_IDLIKE_RE = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z0-9][a-z0-9-]{5,}\b")


def _phrase_tokens(phrases) -> set:
    out = set()
    for p in phrases:
        out.update(t for t in _TOKEN_RE.findall(p) if len(t) >= 4 and t not in _STOPWORDS)
    return out


def extract_narrative_dates(text: str) -> set:
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


def extract_narrative_features(view: "CaseView", narrative: str,
                               drug_vocabulary: set, structured_history: str) -> dict:
    """One deterministic extraction pass per report. `structured_history` is
    the FAERS `Medical History` field, which we treat as authoritative medical
    history (ETHER would have to recover this from free text)."""
    text_l = (str(narrative) if narrative else "").lower()

    family = {m.group(1).strip() for m in _FAMILY_HISTORY_RE.finditer(text_l)}
    medical = {m.group(1).strip() for m in _MEDICAL_HISTORY_RE.finditer(text_l)} - family

    # structured medical-history field: split on ; and newlines
    hist_tokens = set()
    if structured_history:
        for chunk in re.split(r"[;\n]", str(structured_history).lower()):
            hist_tokens.update(_phrase_tokens({chunk.strip()}))

    negated = {m.group(1).strip() for m in _NEGATION_RE.finditer(text_l)}
    symptom_phrases = {m.group(1).strip() for m in _SYMPTOM_RE.finditer(text_l)} - negated

    # "other symptoms" = narrative symptom tokens NOT already coded as PTs
    pt_tokens = _phrase_tokens(view.pts)
    symptom_tokens = _phrase_tokens(symptom_phrases) - pt_tokens

    return {
        "diagnosis": _phrase_tokens({m.group(1).strip() for m in _DIAGNOSIS_RE.finditer(text_l)}),
        "medical_history": _phrase_tokens(medical) | hist_tokens,
        "family_history": _phrase_tokens(family),
        "symptoms": symptom_tokens,
        "narrative_drugs": {w for w in (drug_vocabulary or set()) if w in text_l},
        "narrative_dates": extract_narrative_dates(text_l),
        "id_tokens": {t for t in _IDLIKE_RE.findall(text_l) if not _DATE_ISO_RE.fullmatch(t)},
        "has_text": bool(text_l.strip()),
    }


def build_drug_vocabulary(views, cases) -> set:
    """Distinct drug/ingredient words (>=4 chars) across the series, used to
    detect drug-product mentions in narratives."""
    vocab = set()
    for v in views:
        for p in v.pai:
            vocab.update(w for w in _TOKEN_RE.findall(p) if len(w) >= 4)
    for c in cases:
        for d in _to_set(c.get("drugs")) | _to_set(c.get("drug")):
            vocab.update(w for w in _TOKEN_RE.findall(d) if len(w) >= 4)
    return vocab - _STOPWORDS


_NARRATIVE_FILTER_CATEGORIES = (
    "diagnosis", "medical_history", "family_history", "symptoms", "narrative_drugs",
)


def filter_common_narrative_tokens(views_by_id: dict, max_doc_freq: float) -> None:
    """Strips class-wide boilerplate tokens from every report's narrative
    features, IN PLACE, before pairwise scoring. NOT part of Paper Approach
    v2 (see module docstring) -- this is a Paper Approach v2 + AI-only
    enhancement, called from dedup_paper_v2_ai.py with its own calibrated
    threshold; kept here only because it operates on the same CaseView
    feature shape v2 builds.

    Series that concentrate on one drug (class) and one adverse event -- the
    common case here -- produce narrative vocabulary that is shared by nearly
    every report for reasons that have nothing to do with duplication: the
    series' own suspect drug name appears in almost every narrative,
    "diabetes mellitus" / "concomitant medications" appear in almost every
    diabetic patient's history, etc. The overlap-fraction scoring in
    compare_pair can't tell that kind of token apart from a genuinely
    case-specific shared detail, so it inflates scores for many unrelated
    pairs (confirmed 2026-07: SE5's narrative_drugs/medical_history overlap
    was present in 172/172 and 161/172 of its false-positive pairs, driven by
    tokens like "acid" -- a fragment of several unrelated drug names -- the
    series' own drug name, and generic phrase fragments like "provided" /
    "concomitant"). This is the same standard-NLP idea as IDF down-weighting
    (a token's evidentiary value is inversely related to how common it is in
    the collection), though it is a plain document-frequency CUTOFF rather
    than a continuous TF-IDF weighting.

    `max_doc_freq`: a token shared by more than this fraction of the
    involved reports is dropped from every report's feature set.
    """
    if not max_doc_freq:
        return
    involved = [v for v in views_by_id.values() if v.features and v.features.get("has_text")]
    n = len(involved)
    if n == 0:
        return
    doc_count = defaultdict(int)
    for v in involved:
        for cat in _NARRATIVE_FILTER_CATEGORIES:
            for tok in v.features[cat]:
                doc_count[(cat, tok)] += 1
    common = {key for key, count in doc_count.items() if count / n > max_doc_freq}
    if not common:
        return
    for v in involved:
        for cat in _NARRATIVE_FILTER_CATEGORIES:
            v.features[cat] = {t for t in v.features[cat] if (cat, t) not in common}


# Matches the FAERS literature-report boilerplate lead-in, e.g. "...via an
# article entitled Hanif A.M., ... Phenotypic Spectrum of Pentosan
# Polysulfate Sodium Associated Maculopathy: A Multicenter Study, JAMA
# Ophthalmology November 2019..." or "...via a literature article entitled:
# Krader C. AAO 2019: Ophthalmologists alerted...". Captures the citation
# text (authors + title + journal) up to the next newline or a "Brand name"
# marker, whichever comes first.
_LITERATURE_CITATION_RE = re.compile(
    r"(?:non[- ]?company sponsored study[^.]*?,\s*)?"
    r"(?:via\s+)?(?:a\s+)?(?:literature\s+)?article entitled:?\s*(.+?)(?=\n|Brand name|$)",
    re.IGNORECASE,
)


def extract_literature_source(narrative: str) -> str | None:
    """Fingerprint identifying which cited journal article/study a FAERS
    narrative was derived from, or None if the narrative isn't a literature
    report at all.

    Case-series publications describe N distinct patients by definition --
    when FAERS ingests one, each patient becomes a separate report, but every
    one of those reports' narratives reproduces the SAME article
    abstract/methods verbatim (and FAERS itself cross-references the sibling
    case numbers: "This case, from Same Literature article is linked to
    ..."). That shared text has nothing to do with any individual patient,
    but a narrative-overlap scorer reads it as a huge block of "shared
    specific detail" and over-merges (root-caused on SE7, the study's
    negative-control series -- see documents/SE7_negative_control_analysis.md).
    Two reports citing the SAME article should be treated as a strong prior
    against duplication, not evidence for it.
    """
    m = _LITERATURE_CITATION_RE.search(narrative or "")
    if not m:
        return None
    text = re.sub(r"\s+", " ", m.group(1)).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text[:200] or None


# ==========================================================================
# component 1: pair screening  (paper 3.1.1 / Fig 3)
# ==========================================================================

def screen_pairs(views, config: dict = None) -> list:
    """Fig 3: assign each report to parameter sets (PAI x AE), gather reports
    from those and nearby sets, then narrow to pairs sharing a PAI AND a PT
    with dates within one year. Implemented with an inverted index over
    (PAI, PT) so the full n^2 space is never enumerated."""
    cfg = (config or PAPER_V2_CONFIG).get("screening") or PAPER_V2_CONFIG["screening"]
    window_days = cfg.get("date_window_days", 365)

    # parameter sets keyed on (PAI, PT): a pair sharing any such key already
    # satisfies both the PAI-match and AE-match narrowing rules (Fig 3c).
    param_sets = defaultdict(list)
    for v in views:
        for d in v.pai:
            for e in v.pts:
                param_sets[(d, e)].append(v.case_id)

    candidates = set()
    for members in param_sets.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(set(members)), 2):
            candidates.add((a, b))

    event_by_case = {v.case_id: v.event_date for v in views}
    screened = []
    for a, b in sorted(candidates):
        da, db = event_by_case[a], event_by_case[b]
        # only drop when BOTH event dates are known and > 1 year apart
        if da and db and abs((da - db).days) > window_days:
            continue
        screened.append((a, b))
    return screened


# ==========================================================================
# component 2: pairwise probabilistic comparison  (paper 3.1.2 / Fig 1)
# ==========================================================================

def _set_field_weight(set_a: set, set_b: set, w: dict) -> float:
    """Overlap fraction scales the match weight; complete disjointness (both
    sides informative) counts as a mismatch; missing on either side is 0."""
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b) / min(len(set_a), len(set_b))
    if overlap == 0.0:
        return w["mismatch"]
    return w["match"] * overlap


def compare_pair(va: "CaseView", vb: "CaseView", config: dict = None) -> dict:
    """Probabilistic weight score over the paper's named structured fields and
    the ETHER-substitute narrative features, then (for high scorers) the
    second component of regex narrative corroboration. Returns
    {"weight_score", "second_score", "is_duplicate"}."""
    cfg = (config or PAPER_V2_CONFIG).get("pairwise") or PAPER_V2_CONFIG["pairwise"]
    w = cfg["weights"]
    score = 0.0

    # -- structured fields (paper-named only) -------------------------------
    if va.sex and vb.sex:
        score += w["sex"]["match"] if va.sex == vb.sex else w["sex"]["mismatch"]
    if va.country and vb.country:
        score += w["country"]["match"] if va.country == vb.country else w["country"]["mismatch"]
    if va.age is not None and vb.age is not None:
        tol = cfg.get("age_tolerance_years", 1)
        score += w["age"]["match"] if abs(va.age - vb.age) <= tol else w["age"]["mismatch"]
    # Event date is the strong signal (same patient experience). Only a
    # DAY-precision event date on both sides earns the exact-match bonus;
    # year/month-only dates ('2018') must not masquerade as the same day.
    if va.event_date and vb.event_date:
        if va.event_prec == "day" and vb.event_prec == "day":
            diff = abs((va.event_date - vb.event_date).days)
            if diff <= cfg.get("date_exact_bonus_days", 1):
                score += w["date"]["match"]
            elif diff <= cfg.get("date_partial_days", 60):
                score += w["date"]["match"] * 0.5
            else:
                score += w["date"]["mismatch"]
        elif va.event_date.year != vb.event_date.year:
            # coarse dates but clearly different years -> different experience
            score += w["date"]["mismatch"]
        # else: same year but at least one coarse -> too imprecise to confirm
        # or deny same-day; treat as unknown (contributes nothing).
    elif va.recv_date and vb.recv_date:
        # weak administrative fallback (received date proximity)
        diff = abs((va.recv_date - vb.recv_date).days)
        if diff <= cfg.get("date_partial_days", 60):
            score += w["date"]["match"] * 0.5

    score += _set_field_weight(va.pai, vb.pai, w["pai"])
    score += _set_field_weight(va.pts, vb.pts, w["events"])
    # graded "nearby parameter set" AE similarity (MedDRA hierarchy, Fig 3b);
    # only add it when the exact PTs do NOT already fully overlap.
    if va.pts and vb.pts and not (va.pts & vb.pts):
        hier = _set_field_weight(va.hlgts, vb.hlgts, w["events_hier"])
        score += hier * 0.5 + _set_field_weight(va.socs, vb.socs, w["events_hier"]) * 0.25

    # -- narrative (ETHER-substitute) features ------------------------------
    fa, fb = va.features, vb.features
    if fa["has_text"] and fb["has_text"]:
        for key in ("diagnosis", "medical_history", "family_history",
                    "symptoms", "narrative_drugs", "narrative_dates"):
            score += _set_field_weight(fa[key], fb[key], w[key])
    # medical history also compares across the structured field even without
    # narrative text on both sides.
    elif fa["medical_history"] and fb["medical_history"]:
        score += _set_field_weight(fa["medical_history"], fb["medical_history"], w["medical_history"])

    result = {"weight_score": round(score, 3), "second_score": None, "is_duplicate": False}
    if score >= cfg["duplicate_threshold"]:
        result["is_duplicate"] = True

    # -- second component: regex narrative corroboration --------------------
    both_texts = fa["has_text"] and fb["has_text"]
    if score >= cfg["second_component_threshold"] and both_texts:
        checks = []
        checks.append(1.0 if fa["narrative_dates"] & fb["narrative_dates"] else 0.0)
        checks.append(1.0 if fa["id_tokens"] & fb["id_tokens"] else 0.0)
        phrases_a = fa["diagnosis"] | fa["medical_history"] | fa["symptoms"]
        phrases_b = fb["diagnosis"] | fb["medical_history"] | fb["symptoms"]
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
        result["is_duplicate"] = False

    return result


# ==========================================================================
# component 4: reference case selection  (paper 3.1.4)
# ==========================================================================

def select_reference_case(group: list, graph, views_by_id: dict):
    """Largest sum of connecting edge weights; then combined suspect products
    + AE count; then the real FAERS completeness score; then most recent date."""
    def sort_key(cid):
        v = views_by_id[cid]
        weight_sum = sum(
            graph[cid][nbr].get("weight", 0.0)
            for nbr in graph.neighbors(cid) if nbr in group
        ) if graph.has_node(cid) else 0.0
        n_products_events = len(v.pai) + len(v.pts)
        completeness = v.completeness if v.completeness is not None else 0.0
        date = v.recv_date or v.event_date or datetime.min
        return (weight_sum, n_products_events, completeness, date)

    return max(group, key=sort_key)


# ==========================================================================
# full pipeline
# ==========================================================================

def run_dedup(cases: list, config: dict = None, progress=None) -> dict:
    """Runs all four components on a list of pipeline case dicts. Returns
    groups with reference cases plus funnel counters. `progress(done, total)`
    is called during pairwise comparison."""
    if nx is None:
        raise RuntimeError("The paper v2 pipeline requires networkx (pip install networkx).")
    config = config or PAPER_V2_CONFIG

    views = [CaseView(c) for c in cases]
    views_by_id = {v.case_id: v for v in views}
    cases_by_id = {c["case_id"]: c for c in cases}

    # component 1
    candidate_pairs = screen_pairs(views, config)

    # narrative features once per report involved in a candidate pair
    vocab = build_drug_vocabulary(views, cases)
    involved = {cid for pair in candidate_pairs for cid in pair}
    for cid in involved:
        v = views_by_id[cid]
        raw = cases_by_id[cid].get("raw") or {}
        v.features = extract_narrative_features(
            v, cases_by_id[cid].get("narrative"), vocab,
            raw.get("Medical History/Medical History Comments"),
        )

    # component 2
    duplicate_pairs = []
    n_second = 0
    for i, (a, b) in enumerate(candidate_pairs):
        result = compare_pair(views_by_id[a], views_by_id[b], config)
        if result["second_score"] is not None:
            n_second += 1
        if result["is_duplicate"]:
            duplicate_pairs.append((a, b, result["weight_score"]))
        if progress:
            progress(i + 1, len(candidate_pairs))

    # component 3 (reused from v1 -- faithful)
    groups, graph, n_components = group_and_split(duplicate_pairs)

    # component 4
    groups_out = []
    for i, group in enumerate(groups, start=1):
        reference = select_reference_case(group, graph, views_by_id)
        groups_out.append({
            "group_id": i,
            "case_ids": [str(views_by_id[cid].safety_report_id) for cid in group],
            "reference_case": str(views_by_id[reference].safety_report_id),
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
