"""
Orchestration for the standalone Dedup Study app -- the research subset of
the main askMyFAERS dedup pipeline, on the standalone SQLite store:

- parse_cases_payload / replace_series_cases: flexible CSV/XLSX/JSON case
  ingestion (one row per safety_report_id).
- run_bucketing: Skill A (AI-free), always a clean restart under the active
  skill version; singleton buckets stay untagged.
- run_analysis: Skill B tiers 2 / 2.5 / 3 across every bucket, then freezes
  the result as an immutable snapshot auto-scored against the active ground
  truth. One result per skill version is shown (latest run supersedes),
  history stays in the snapshots.
- run_batch_audit: ONE LLM call per discrepant group of the latest scored
  snapshot (concise per-case verdict/cause/clue), persisted per case.
- generate_refine_strategy: one LLM call synthesizing all audits into a
  Markdown improvement plan.

Simplifications vs the main app (research tool, not a review workflow):
no manual tag editing, no confirm/dismiss human decisions, no version
lineages (uploads are unique per safety_report_id), every analysis is a
fresh full run.
"""

import io
import json
import logging
import math
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import ai_client
import dedup_core
import dedup_metrics
import dedup_paper
import dedup_paper_v2
import dedup_paper_v2_ai
import prompts
from db import connect, j, unj, get_setting
from versioning import get_active_config

logger = logging.getLogger(__name__)

MAX_COMPANIONS = 6
COMPANION_NARRATIVE_CHARS = 1500
MAX_GROUP_FOCALS = 10
FOCAL_NARRATIVE_CHARS = 3000

# Case-insensitive header aliases for the uploaded case table.
COLUMN_ALIASES = {
    "safety_report_id": ["safety_report_id", "safetyreportid", "case_id", "caseid", "case id",
                         "report_id", "report id", "case number", "case_number", "id",
                         "faers case #", "faers case number", "case #", "isr number", "isr"],
    "age": ["age", "patient age", "patientonsetage", "age_years", "age in years", "age_in_years"],
    "sex": ["sex", "gender", "patient sex", "patientsex"],
    "country": ["country", "occurcountry", "reporter country", "country_derived", "occur_country",
                "country derived"],
    "initial_date": ["initial_date", "receive date", "receivedate", "report date", "report_date",
                     "date", "initialdate", "initial date", "initial fda received date",
                     "latest fda received date"],
    "drug": ["drug", "suspect_drug", "suspect drug", "drugname", "medicinalproduct", "primary_drug"],
    "event": ["event", "adverse_event", "adverse event", "reaction", "pt", "primary_event"],
    "drugs": ["drugs", "drug_list", "all_drugs", "all suspect product names", "primary suspect (ps)"],
    "events": ["events", "event_list", "all_events", "reactions", "pts", "all pts"],
    "narrative": ["narrative", "case narrative", "case_narrative", "description", "text", "case_text"],
}


class PipelineError(Exception):
    """User-facing precondition failure (maps to a 4xx at the API layer)."""


def ai_settings() -> dict:
    """Resolved call settings for the model option selected in the UI (one of
    the four fixed options; connection details come from .env)."""
    option = (get_setting("ai") or {}).get("model_option") or ai_client.DEFAULT_MODEL_OPTION
    mock_mode = get_setting("mock_mode", False)
    try:
        return ai_client.resolve_option(option, bypass_configured=mock_mode)
    except ai_client.AIError as e:
        raise PipelineError(str(e))


# --------------------------------------------------------------------------
# Case ingestion
# --------------------------------------------------------------------------

def _split_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    text = str(value)
    for sep in ("|", ";", "\n"):
        if sep in text:
            return sorted({p.strip().lower() for p in text.split(sep) if p.strip()})
    return [text.strip().lower()] if text.strip() else []


def _match_columns(headers: list) -> dict:
    """header -> canonical field, via case-insensitive alias matching."""
    mapping = {}
    lowered = {str(h).strip().lower(): h for h in headers}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered and field not in mapping.values():
                mapping[lowered[alias]] = field
                break
    return mapping


def clean_json_value(value):
    """JSON-safe scrub, recursive: NaN/Inf floats and 'nan'/'nat' strings
    become None. Pandas keeps NaN in float columns even after
    .where(pd.notna(...), None), and bare NaN in a JSON payload is invalid --
    the browser's JSON.parse rejects the whole response, which is how
    demographics ended up rendering as em-dashes."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, str) and value.strip().lower() in ("nan", "nat"):
        return None
    if isinstance(value, dict):
        return {k: clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json_value(v) for v in value]
    return value


def _row_to_case(row: dict, mapping: dict) -> dict:
    # Scrub NaN up front so every "in (None, '')" check below also skips
    # missing float cells, and str(value) can never become the string "nan".
    row = {k: clean_json_value(v) for k, v in row.items()}
    case = {"raw": {k: v for k, v in row.items() if v not in (None, "")}}
    for header, field in mapping.items():
        value = row.get(header)
        if value in (None, ""):
            continue
        if field in ("drugs", "events"):
            case[field] = _split_list(value)
        else:
            case[field] = str(value).strip()

    # Auto-extract drugs from multiple Product columns
    extracted_drugs = set()
    if case.get("drugs"):
        extracted_drugs.update(case["drugs"])
    if case.get("drug"):
        extracted_drugs.update(_split_list(case["drug"]))

    for k, v in row.items():
        if v in (None, ""):
            continue
        kl = str(k).lower().strip()
        if re.search(r"product \d+ product name", kl) or re.search(r"product \d+ product active ingredient", kl):
            extracted_drugs.update(_split_list(v))

    if extracted_drugs:
        case["drugs"] = sorted(list(extracted_drugs))
        if not case.get("drug") and case["drugs"]:
            case["drug"] = case["drugs"][0]

    # Auto-extract events from multiple PT Term Event columns
    extracted_events = set()
    if case.get("events"):
        extracted_events.update(case["events"])
    if case.get("event"):
        extracted_events.update(_split_list(case["event"]))

    for k, v in row.items():
        if v in (None, ""):
            continue
        kl = str(k).lower().strip()
        if re.search(r"pt term event \d+", kl):
            extracted_events.update(_split_list(v))

    if extracted_events:
        case["events"] = sorted(list(extracted_events))
        if not case.get("event") and case["events"]:
            case["event"] = case["events"][0]

    return case


def parse_cases_payload(filename: str, content: bytes) -> list:
    """Parses an uploaded CSV / XLSX / JSON file into case dicts. JSON is a
    list of objects keyed however the source keyed them (aliases apply)."""
    name = (filename or "").lower()
    rows = []
    if name.endswith(".json"):
        parsed = json.loads(content.decode("utf-8-sig"))
        if isinstance(parsed, dict) and isinstance(parsed.get("cases"), list):
            parsed = parsed["cases"]
        if not isinstance(parsed, list):
            raise PipelineError("JSON case files must be a list of case objects (or {\"cases\": [...]}).")
        rows = [r for r in parsed if isinstance(r, dict)]
    else:
        try:
            import pandas as pd
        except ImportError:
            raise PipelineError("Reading CSV/Excel files requires pandas (pip install pandas openpyxl).")
        buffer = io.BytesIO(content)
        if name.endswith((".xlsx", ".xls", ".xlsm")):
            xl = pd.ExcelFile(buffer)
            sheet_name = xl.sheet_names[0]
            for sname in xl.sheet_names:
                if sname.strip().lower() == "case details":
                    sheet_name = sname
                    break
            
            df_temp = pd.read_excel(xl, sheet_name=sheet_name, header=None, nrows=15)
            header_row_idx = 0
            id_aliases = set(COLUMN_ALIASES["safety_report_id"])
            for idx, row in df_temp.iterrows():
                row_vals = [str(v).strip().lower() for v in row if v not in (None, "")]
                if any(val in id_aliases for val in row_vals):
                    header_row_idx = idx
                    break
            
            buffer.seek(0)
            df = pd.read_excel(buffer, sheet_name=sheet_name, header=header_row_idx)
        else:
            df_temp = pd.read_csv(buffer, header=None, nrows=15)
            header_row_idx = 0
            id_aliases = set(COLUMN_ALIASES["safety_report_id"])
            for idx, row in df_temp.iterrows():
                row_vals = [str(v).strip().lower() for v in row if v not in (None, "")]
                if any(val in id_aliases for val in row_vals):
                    header_row_idx = idx
                    break
            
            buffer.seek(0)
            df = pd.read_csv(buffer, header=header_row_idx)
        df = df.where(pd.notna(df), None)
        rows = df.to_dict(orient="records")

    if not rows:
        raise PipelineError("No rows found in the uploaded file.")

    mapping = _match_columns(rows[0].keys())
    if "safety_report_id" not in mapping.values():
        raise PipelineError(
            "No case-id column found. Expected one of: "
            + ", ".join(COLUMN_ALIASES["safety_report_id"])
        )

    cases = []
    seen = set()
    for row in rows:
        case = _row_to_case(row, mapping)
        report_id = case.get("safety_report_id")
        if not report_id or report_id in seen:
            continue  # duplicate report ids in one upload: first row wins
        seen.add(report_id)
        cases.append(case)
    if not cases:
        raise PipelineError("No usable cases (every row lacked a case id).")
    return cases


def replace_series_cases(series_id: int, cases: list) -> int:
    """Replaces the series' cases with the upload (also clearing bucket
    state, since the case universe changed). Snapshots/audits are immutable
    history and stay."""
    with connect() as conn:
        conn.execute("DELETE FROM cases WHERE series_id = ?", (series_id,))
        for c in cases:
            conn.execute(
                "INSERT INTO cases(series_id, safety_report_id, age, sex, country, initial_date, "
                "drug, event, drugs, events, narrative, raw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (series_id, c.get("safety_report_id"), c.get("age"), c.get("sex"), c.get("country"),
                 c.get("initial_date"), c.get("drug"), c.get("event"),
                 j(c.get("drugs") or []), j(c.get("events") or []),
                 c.get("narrative"), j(c.get("raw") or {})),
            )
        conn.execute("UPDATE series SET bucketing_version = NULL WHERE id = ?", (series_id,))
    return len(cases)


def load_series_cases(series_id: int) -> list:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM cases WHERE series_id = ? ORDER BY id", (series_id,)).fetchall()
    # clean_json_value also covers rows uploaded BEFORE the NaN scrub existed:
    # their age/raw columns still hold 'nan' strings / NaN floats in the DB.
    return [
        {
            "case_id": r["id"],
            "safety_report_id": r["safety_report_id"],
            "age": clean_json_value(r["age"]), "sex": clean_json_value(r["sex"]),
            "country": clean_json_value(r["country"]),
            "initial_date": clean_json_value(r["initial_date"]),
            "drug": clean_json_value(r["drug"]), "event": clean_json_value(r["event"]),
            "drugs": clean_json_value(unj(r["drugs"], [])), "events": clean_json_value(unj(r["events"], [])),
            "narrative": r["narrative"] or "",
            "raw": clean_json_value(unj(r["raw"], {})),
            "tag": r["tag"],
            "ai_summary": unj(r["ai_summary"], {}),
        }
        for r in rows
    ]


# --------------------------------------------------------------------------
# Skill A: bucketing (always reset + re-run under the active version)
# --------------------------------------------------------------------------

def _summarize_case(case: dict, config: dict, settings: dict) -> tuple:
    """Executes the AI Case Summarizer on a single case.
    If mock mode is active, generates a mock JSON summary.
    Returns (summary_dict, usage_dict)."""
    if get_setting("mock_mode", False):
        # Local regex to extract potential lot numbers even in mock mode
        narrative = case.get("narrative") or ""
        match = re.search(r'(?:lot|batch)\s*(?:#|no\.?)?\s*([A-Za-z0-9-]+)', narrative, re.IGNORECASE)
        lot = match.group(1).upper() if match else None
        
        # Infer role
        role = "patient"
        n_low = narrative.lower()
        if any(w in n_low for w in ("nurse", "doctor", "physician", "pharmacist", "healthcare worker", "hcw")):
            role = "healthcare_worker"
        elif any(w in n_low for w in ("parent", "mother", "father", "consumer", "caregiver")):
            role = "consumer"
            
        mock_summary = {
            "lot_number": lot,
            "serial_number": None,
            "patient_role": role,
            "primary_adverse_event": case.get("event") or "adverse event",
            "specific_mechanism_or_circumstance": "Mock extracted mechanism from narrative",
            "journal_reference": "Mock Journal 2026" if "pmid" in n_low or "doi" in n_low or "journal" in n_low else None
        }
        return mock_summary, {"input_tokens": 100, "output_tokens": 50, "cost": 0.0001}

    # Call LLM
    system_instructions = prompts.DEDUP_SUMMARIZATION_SYSTEM_INSTRUCTIONS
    user_prompt = prompts.DEDUP_SUMMARIZATION_USER_PROMPT
    
    payload = {
        k: case.get(k) for k in
        ("safety_report_id", "age", "sex", "country", "initial_date", "drugs", "events", "narrative")
    }
    
    user_message = user_prompt.replace("{data}", json.dumps(payload, default=str))
    full_prompt = system_instructions + "\n\n" + user_message
    
    try:
        text, usage = ai_client.generate(full_prompt, settings)
        parsed = ai_client.parse_json_response(text)
    except Exception as e:
        logger.error(f"AI Case Summarization failed for case {case['safety_report_id']}: {e}")
        return {}, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        
    return parsed, usage


def ensure_ai_summaries(series_id: int, config: dict) -> list:
    """Loads cases, runs TF-IDF pre-filter, executes parallel AI summarization
    on candidates lacking summaries, caches them in the DB, and returns the list
    of all cases (with ai_summary populated on candidates)."""
    cases = load_series_cases(series_id)
    if not cases:
        return []
        
    bucketing_cfg = (config or {}).get("bucketing") or {}
    is_smart = bucketing_cfg.get("type") == "smart"
    if not is_smart:
        return cases

    # 1. TF-IDF & Drug Pre-filtering to find candidates vs singletons
    candidates = []
    if len(cases) > 1:
        narratives = {c["case_id"]: c.get("narrative") or "" for c in cases}
        vectors = dedup_core._tfidf_vectors(narratives)
        prefilter_threshold = bucketing_cfg.get("prefilter_threshold") or 0.12
        
        case_drugs = {c["case_id"]: set(dedup_core._to_set(c.get("drugs"))) for c in cases}
        
        for a in cases:
            cid_a = a["case_id"]
            drugs_a = case_drugs[cid_a]
            has_potential_partner = False
            
            for b in cases:
                cid_b = b["case_id"]
                if cid_a == cid_b:
                    continue
                
                # Check 1: Demographic similarity
                dem_a = dedup_core.compute_case_tag(a, config)
                dem_b = dedup_core.compute_case_tag(b, config)
                if dem_a and dem_b and dem_a == dem_b:
                    has_potential_partner = True
                    break
                    
                # Check 2: Drug overlap + narrative similarity
                drugs_b = case_drugs[cid_b]
                if drugs_a & drugs_b:
                    sim = dedup_core._cosine_similarity(vectors.get(cid_a, {}), vectors.get(cid_b, {}))
                    if sim >= prefilter_threshold:
                        has_potential_partner = True
                        break
                        
            if has_potential_partner:
                candidates.append(a)
    else:
        candidates = cases

    # 2. AI Summarize candidates lacking cached summaries
    to_summarize = [c for c in candidates if not c.get("ai_summary")]
    if to_summarize:
        settings = ai_settings()
        summaries = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_case = {
                executor.submit(_summarize_case, c, config, settings): c
                for c in to_summarize
            }
            for future in as_completed(future_to_case):
                c = future_to_case[future]
                try:
                    summary, usage = future.result()
                    summaries[c["case_id"]] = summary
                except Exception as e:
                    logger.error(f"Error in parallel summarization worker for case {c['safety_report_id']}: {e}")
                    summaries[c["case_id"]] = {}
                    
        with connect() as conn:
            for cid, summary in summaries.items():
                conn.execute("UPDATE cases SET ai_summary = ? WHERE id = ?", (j(summary), cid))
                
        # Reload newly generated summaries in memory
        for c in cases:
            if c["case_id"] in summaries:
                c["ai_summary"] = summaries[c["case_id"]]
                
    return cases


def run_bucketing(series_id: int) -> dict:
    version, config = get_active_config()
    cases = ensure_ai_summaries(series_id, config)
    if not cases:
        raise PipelineError("The series has no cases. Upload a case file first.")
    
    # Smart or standard tagging
    tags = dedup_core.run_smart_tagging(cases, config=config)
    with connect() as conn:
        conn.execute("UPDATE cases SET tag = NULL WHERE series_id = ?", (series_id,))
        for case_id, tag in tags.items():
            conn.execute("UPDATE cases SET tag = ? WHERE id = ?", (tag, case_id))
        conn.execute("UPDATE series SET bucketing_version = ? WHERE id = ?", (version, series_id))
    return {
        "cases_tagged": len(tags),
        "bucket_count": len(set(tags.values())),
        "skill_version": version,
    }


# --------------------------------------------------------------------------
# Skill B: tiered analysis + snapshot
# --------------------------------------------------------------------------

def _narrative_compare(case_a: dict, case_b: dict, config: dict, settings: dict) -> tuple:
    """Tier 3: one LLM call judging whether two cases are the same
    real-world case. Errors degrade to an 'uncertain' verdict.
    Returns (result_dict, usage_dict)."""
    if get_setting("mock_mode", False):
        import random
        # check Jaccard similarity of suspect drugs
        set_a_drugs = set(case_a.get("drugs") or [])
        set_b_drugs = set(case_b.get("drugs") or [])
        drug_jac = len(set_a_drugs & set_b_drugs) / max(len(set_a_drugs | set_b_drugs), 1)
        
        # If drug similarity is high, let's mock a duplicate
        if drug_jac > 0.5:
            verdict = "same"
            confidence = "high" if drug_jac > 0.8 else "medium"
            reasoning = f"[MOCK] High drug overlap ({drug_jac:.2f}). Demographics match or are compatible."
        else:
            verdict = "different"
            confidence = "high"
            reasoning = f"[MOCK] Low drug overlap ({drug_jac:.2f})."
            
        mock_parsed = {
            "verdict": verdict,
            "confidence": confidence,
            "shared_specific_detail": "Mock overlap of suspect drugs",
            "reasoning": reasoning
        }
        mock_usage = {"input_tokens": 120, "output_tokens": 60, "cost": 0.00018}
        return mock_parsed, mock_usage

    narrative_cfg = (config or {}).get("narrative_llm") or {}
    system_instructions = narrative_cfg.get("system_instructions") or prompts.DEDUP_NARRATIVE_SYSTEM_INSTRUCTIONS
    user_prompt = narrative_cfg.get("user_prompt") or prompts.DEDUP_NARRATIVE_USER_PROMPT

    def payload(case):
        return case.get("raw") or {
            k: case.get(k) for k in
            ("safety_report_id", "age", "sex", "country", "initial_date", "drugs", "events", "narrative")
        }

    user_message = (
        user_prompt
        .replace("{safety_report_id_a}", str(case_a["safety_report_id"]))
        .replace("{safety_report_id_b}", str(case_b["safety_report_id"]))
        .replace("{data_a}", json.dumps(payload(case_a), default=str))
        .replace("{data_b}", json.dumps(payload(case_b), default=str))
    )
    full_prompt = system_instructions + "\n\n" + user_message
    try:
        text, usage = ai_client.generate(full_prompt, settings)
        parsed = ai_client.parse_json_response(text)
    except Exception as e:
        logger.error(f"Tier-3 comparison failed ({case_a['safety_report_id']}, {case_b['safety_report_id']}): {e}")
        return {"verdict": "uncertain", "confidence": "low", "reasoning": f"AI error: {e}", "error": str(e)}, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    return {
        "verdict": parsed.get("verdict", "uncertain"),
        "confidence": parsed.get("confidence", "low"),
        "shared_specific_detail": parsed.get("shared_specific_detail", ""),
        "reasoning": parsed.get("reasoning", ""),
    }, usage


def run_analysis(series_id: int, task=None) -> dict:
    """Skill B across every bucket in the series, then an immutable snapshot
    auto-scored against the active ground truth. Progress ticks per ambiguous
    pair (each is a potential LLM call)."""
    started_at = datetime.utcnow()
    version, config = get_active_config()
    settings = ai_settings()

    # Map new schema to legacy keys for compatibility
    pairwise = config.get("pairwise") or {}
    weights = pairwise.get("weights") or {}
    config = {
        **config,
        "bucketing": config.get("bucketing") or {
            "fields": ["age_band", "sex", "country"],
            "age_band_years": 5,
        },
        "scoring": config.get("scoring") or {
            "high_confidence_threshold": 0.85,
            "ambiguous_threshold": 0.55,
            "weights": {
                "sex": weights.get("sex", {}).get("match", 1.0),
                "country": weights.get("country", {}).get("match", 1.0),
                "age": weights.get("age", {}).get("match", 1.0),
                "drugs": weights.get("drugs", {}).get("match", 2.0),
                "events": weights.get("events", {}).get("match", 2.0),
                "date": weights.get("date", {}).get("match", 1.0),
            },
            "age_tolerance_years": pairwise.get("age_tolerance_years", 10),
            "date_tolerance_days": pairwise.get("date_partial_days", 30),
            "max_group_size": 60,
            "clustering_algorithm": "union_find",
            "fallback_narrative_threshold": 0.75,
        },
        "narrative_llm": config.get("narrative_llm") or {
            "merge_confidence": ["high", "medium"],
            "system_instructions": prompts.DEDUP_NARRATIVE_SYSTEM_INSTRUCTIONS,
            "user_prompt": prompts.DEDUP_NARRATIVE_USER_PROMPT,
        }
    }

    with connect() as conn:
        series = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if series is None:
        raise PipelineError("Series not found.")
    if series["bucketing_version"] is None or series["bucketing_version"] != version:
        run_bucketing(series_id)

    cases = ensure_ai_summaries(series_id, config)
    by_id = {c["case_id"]: c for c in cases}

    # 1. Compute standard tags and build initial buckets
    standard_tags = dedup_core.run_smart_tagging(cases, config=config)
    for c in cases:
        c["tag"] = standard_tags.get(c["case_id"])

    # 2. Find fallback candidate pairs using the new date, MFR, and narrative similarity rules
    fallback_pairs = []
    
    # Compute TF-IDF vectors for all narratives in the series
    narratives = {c["case_id"]: c.get("narrative") or "" for c in cases}
    vectors = dedup_core._tfidf_vectors(narratives)
    
    fallback_threshold = (config or {}).get("scoring", {}).get("fallback_narrative_threshold") or 0.8
    
    for i, a in enumerate(cases):
        cid_a = a["case_id"]
        mfr_a = dedup_core._get_mfr_control_number(a)
        mfr_core_a = dedup_core.extract_mfr_core(mfr_a) if mfr_a else ""
        
        for b in cases[i+1:]:
            cid_b = b["case_id"]
            mfr_b = dedup_core._get_mfr_control_number(b)
            mfr_core_b = dedup_core.extract_mfr_core(mfr_b) if mfr_b else ""
            
            is_candidate = False
            
            # Scenario A: MFR Core Match
            if mfr_core_a and mfr_core_b and mfr_core_a == mfr_core_b:
                is_candidate = True
                
            # Scenario B: Date Match (meta or narrative date)
            elif dedup_core.check_date_match(a, b):
                is_candidate = True
                
            # Scenario C: High Narrative Similarity
            else:
                sim = dedup_core._cosine_similarity(vectors.get(cid_a, {}), vectors.get(cid_b, {}))
                if sim >= fallback_threshold:
                    is_candidate = True
            
            # Check compatibility before adding
            if is_candidate:
                if dedup_core.check_compatibility(a, b, config):
                    fallback_pairs.append((cid_a, cid_b))

    # 4. Group all cases using standard tags + fallback pairs
    uf_tags = dedup_core.UnionFind([c["case_id"] for c in cases])
    
    # Union standard tag groups
    from collections import defaultdict
    tag_groups = defaultdict(list)
    for c in cases:
        if c["tag"]:
            tag_groups[c["tag"]].append(c["case_id"])
            
    for tag, cids in tag_groups.items():
        for idx in range(1, len(cids)):
            uf_tags.union(cids[0], cids[idx])
            
    # Union fallback pairs
    for cid_a, cid_b in fallback_pairs:
        uf_tags.union(cid_a, cid_b)

    # 5. Extract final buckets
    buckets = {}
    fallback_counter = 1
    for group in uf_tags.groups():
        if len(group) < 2:
            continue
        group_standard_tags = {by_id[cid]["tag"] for cid in group if by_id[cid]["tag"]}
        if group_standard_tags:
            tag_name = sorted(group_standard_tags)[0]
        else:
            tag_name = f"fallback_group_{fallback_counter}"
            fallback_counter += 1
            
        buckets[tag_name] = [by_id[cid] for cid in group]

    if not buckets:
        raise PipelineError("No buckets of size >= 2 found, even after TF-IDF fallback.")

    # Tier 2 / 2.5 for every bucket up front: cheap, and sizes the progress bar.
    funnel = {
        "pairs_scored": 0, "tier2_confirmed_pairs": 0,
        "ambiguous_before_prefilter": 0, "ambiguous_after_prefilter": 0,
        "llm_calls": 0, "tier3_merges": 0,
    }
    plans = []
    total_pairs = 0
    for tag in sorted(buckets):
        members = buckets[tag]
        if len(members) < 2:
            continue
        group_result = dedup_core.run_group_analysis(members, config=config)
        funnel["pairs_scored"] += group_result["pairs_scored"]
        funnel["tier2_confirmed_pairs"] += len(group_result["confirmed_pairs"])
        funnel["ambiguous_before_prefilter"] += group_result["ambiguous_before_prefilter"]
        funnel["ambiguous_after_prefilter"] += len(group_result["ambiguous_pairs"])
        total_pairs += len(group_result["ambiguous_pairs"])
        plans.append((tag, group_result))

    if task:
        task.set_total(total_pairs)

    merge_confidence = set(((config or {}).get("narrative_llm") or {}).get("merge_confidence") or ["high", "medium"])
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    groups_out = []
    cancelled = False

    # Gather tasks to execute in parallel
    tasks_to_run = []
    for tag, group_result in plans:
        uf = group_result["union_find"]
        for a, b, score in group_result["ambiguous_pairs"]:
            tasks_to_run.append((tag, a, b, uf))

    # Initialize bucket similarities for average linkage clustering
    from collections import defaultdict
    bucket_sims = defaultdict(dict)
    for tag, group_result in plans:
        for a, b, score in group_result["confirmed_pairs"]:
            key = tuple(sorted((a, b)))
            bucket_sims[tag][key] = score

    scoring_cfg = (config or {}).get("scoring") or {}
    clustering_algo = scoring_cfg.get("clustering_algorithm") or "union_find"
    ambiguous_threshold = scoring_cfg.get("ambiguous_threshold", dedup_core.AMBIGUOUS_THRESHOLD)

    lock = threading.Lock()

    def process_pair(tag, a, b, uf):
        nonlocal cancelled
        # 1. Thread-safe check of transitive merge
        with lock:
            if cancelled or (task and task.cancelled):
                cancelled = True
                return None
            if clustering_algo == "union_find" and uf.find(a) == uf.find(b):
                return "skipped"

        # 2. Heavy LLM call (lock released, parallel!)
        try:
            result, usage = _narrative_compare(by_id[a], by_id[b], config, settings)
        except Exception as e:
            logger.error(f"Worker narrative comparison exception: {e}")
            result = {"verdict": "uncertain", "confidence": "low", "reasoning": f"Worker error: {e}"}
            usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

        # 3. Thread-safe merge and counter updates
        with lock:
            if cancelled or (task and task.cancelled):
                cancelled = True
                return None

            funnel["llm_calls"] += 1
            usage_total["input_tokens"] += usage.get("input_tokens") or 0
            usage_total["output_tokens"] += usage.get("output_tokens") or 0
            usage_total["cost"] += usage.get("cost") or 0.0

            key = tuple(sorted((a, b)))
            is_same = result.get("verdict") == "same" and result.get("confidence") in merge_confidence
            if is_same:
                bucket_sims[tag][key] = 0.9 if result.get("confidence") == "high" else 0.8
            else:
                bucket_sims[tag][key] = 0.0

            if uf.find(a) != uf.find(b):
                if is_same:
                    uf.union(a, b)
                    funnel["tier3_merges"] += 1

            return "processed"

    if tasks_to_run:
        # Run tasks in parallel
        # Elsa / Ollama can handle some level of concurrency. We cap it at 10 to be polite.
        with ThreadPoolExecutor(max_workers=min(10, len(tasks_to_run))) as executor:
            futures = [
                executor.submit(process_pair, tag, a, b, uf)
                for tag, a, b, uf in tasks_to_run
            ]
            for future in as_completed(futures):
                if task and task.cancelled:
                    cancelled = True
                res = future.result()
                if res is None:
                    cancelled = True
                if task:
                    task.tick()

    # Collect final groupings from each plan
    for tag, group_result in plans:
        uf = group_result["union_find"]
        if clustering_algo == "average_linkage":
            nodes = [m["case_id"] for m in buckets[tag]]
            sims = bucket_sims[tag]
            clusters = dedup_core.average_linkage_clustering(nodes, sims, threshold=ambiguous_threshold)
            for group in clusters:
                if len(group) < 2:
                    continue
                groups_out.append({
                    "group_id": len(groups_out) + 1,
                    "tag": tag,
                    "case_ids": sorted({str(by_id[cid]["safety_report_id"]) for cid in group}),
                })
        elif clustering_algo == "density_split":
            nodes = [m["case_id"] for m in buckets[tag]]
            sims = bucket_sims[tag]
            clusters = dedup_core.split_group_by_density(nodes, sims, edge_threshold=ambiguous_threshold)
            for group in clusters:
                if len(group) < 2:
                    continue
                groups_out.append({
                    "group_id": len(groups_out) + 1,
                    "tag": tag,
                    "case_ids": sorted({str(by_id[cid]["safety_report_id"]) for cid in group}),
                })
        else:
            for group in uf.groups():
                if len(group) < 2:
                    continue
                groups_out.append({
                    "group_id": len(groups_out) + 1,
                    "tag": tag,
                    "case_ids": sorted({str(by_id[cid]["safety_report_id"]) for cid in group}),
                })

    if cancelled:
        # Partial runs are not comparable data points -- record nothing.
        return {"cancelled": True}

    finished_at = datetime.utcnow()

    with connect() as conn:
        gt = conn.execute(
            "SELECT * FROM ground_truths WHERE series_id = ? AND is_active = 1", (series_id,)
        ).fetchone()
    metrics = None
    if gt:
        universe = {str(c["safety_report_id"]) for c in cases}
        metrics = dedup_metrics.compute_metrics(
            [g["case_ids"] for g in groups_out],
            unj(gt["groups"], []),
            universe=universe,
        )

    is_mock_val = 1 if get_setting("mock_mode", False) else 0
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO snapshots(series_id, skill_version, ai_provider, model_name, funnel, result, "
            "ground_truth_id, metrics, started_at, finished_at, duration_seconds, llm_calls, total_tokens, cost, is_mock) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (series_id, version, settings.get("provider"), f"{settings.get('label') or settings.get('model')} (Bucketing + AI Call)",
             j(funnel), j({"groups": groups_out}),
             gt["id"] if gt else None, j(metrics),
             started_at.isoformat(), finished_at.isoformat(),
             int((finished_at - started_at).total_seconds()),
             funnel["llm_calls"],
             usage_total["input_tokens"] + usage_total["output_tokens"],
             round(usage_total["cost"], 6),
             is_mock_val),
        )
        snapshot_id = cursor.lastrowid

    return {
        "snapshot_id": snapshot_id,
        "n_groups": len(groups_out),
        "llm_calls": funnel["llm_calls"],
        "f1": (metrics or {}).get("pair_level", {}).get("f1") if metrics else None,
        "scored": metrics is not None,
    }


# --------------------------------------------------------------------------
# Paper pipeline (Kreimeyer 2025 JBI) -- deterministic, NO AI calls
# --------------------------------------------------------------------------

# Snapshots of the paper pipeline are stored under this pseudo skill version
# so they land in the same comparison matrix as the (currently disabled)
# AI-skill versions without colliding with them.
PAPER_SKILL_VERSION = 0
# Paper approach v2 (revised, higher-fidelity deterministic pipeline;
# dedup_paper_v2.py) is snapshotted under its own pseudo version so it lands
# in the comparison matrix beside v1 ("paper") without colliding with the real
# AI-skill versions (1..N) or v1 (0).
PAPER_V2_SKILL_VERSION = -1
# Enhanced Approach (AI + Narrative-Gated + Doc-Frequency Filter;
# dedup_paper_v2_ai.py) -- built on the deterministic v2 pipeline, with
# AI-assisted narrative extraction swapped in only for the small, gated
# subset of reports where it's likely to matter, plus a document-frequency
# narrative-token filter -- both unique to this approach, absent from the
# published paper (see dedup_paper_v2_ai module docstring). Distinct models
# are distinguished by model_name on the snapshot, same convention as the
# (retired) AI approach.
PAPER_V2_AI_SKILL_VERSION = -2
# Enhanced Approach + Literature-aware override -- a SEPARATE, additional
# approach, not a replacement for PAPER_V2_AI_SKILL_VERSION. Identical to the
# Enhanced Approach except that a pair is forced to non-duplicate whenever
# both reports' narratives cite the same journal article (see
# dedup_paper_v2.extract_literature_source and the SE7 root-cause analysis,
# documents/SE7_negative_control_analysis.md). Supportive/discussion
# evidence for the manuscript, not one of the two main-result approaches --
# those (Enhanced, Scratch) are intentionally left unmodified.
PAPER_V2_AI_LIT_SKILL_VERSION = -3
AI_SKILL_VERSION = 999


def _version_label(skill_version) -> str:
    """Human label for a snapshot/audit pseudo-version."""
    if skill_version == PAPER_SKILL_VERSION:
        return "paper"
    if skill_version == PAPER_V2_SKILL_VERSION:
        return "paperv2"
    if skill_version == PAPER_V2_AI_SKILL_VERSION:
        return "paperv2ai"
    if skill_version == PAPER_V2_AI_LIT_SKILL_VERSION:
        return "paperv2ailit"
    return f"v{skill_version}"


def run_paper_analysis(series_id: int, task=None) -> dict:
    """The four-component InfoViP deduplication pipeline from the paper in
    ./documents (pair screening -> probabilistic pairwise comparison ->
    modularity-based grouping/splitting -> reference case selection),
    entirely deterministic. The result is frozen as a snapshot and scored
    against the active ground truth, exactly like a skill run."""
    started_at = datetime.utcnow()

    with connect() as conn:
        series = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if series is None:
        raise PipelineError("Series not found.")

    cases = load_series_cases(series_id)
    if not cases:
        raise PipelineError("The series has no cases. Upload a case file first.")

    class _Cancelled(Exception):
        pass

    def progress(done, total):
        if task:
            if task.total != total:
                task.set_total(total)
            task.tick()
            if task.cancelled:
                raise _Cancelled()

    try:
        result = dedup_paper.run_dedup(cases, progress=progress)
    except _Cancelled:
        return {"cancelled": True}
    except RuntimeError as e:
        raise PipelineError(str(e))

    groups_out = [
        {"group_id": g["group_id"], "tag": f"paper_group_{g['group_id']}",
         "case_ids": sorted(set(g["case_ids"])), "reference_case": g["reference_case"]}
        for g in result["groups"]
    ]
    funnel = result["funnel"]
    finished_at = datetime.utcnow()

    with connect() as conn:
        gt = conn.execute(
            "SELECT * FROM ground_truths WHERE series_id = ? AND is_active = 1", (series_id,)
        ).fetchone()
    metrics = None
    if gt:
        universe = {str(c["safety_report_id"]) for c in cases}
        metrics = dedup_metrics.compute_metrics(
            [g["case_ids"] for g in groups_out],
            unj(gt["groups"], []),
            universe=universe,
        )

    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO snapshots(series_id, skill_version, ai_provider, model_name, funnel, result, "
            "ground_truth_id, metrics, started_at, finished_at, duration_seconds, llm_calls, total_tokens, cost, is_mock) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (series_id, PAPER_SKILL_VERSION, "none", "InfoViP paper pipeline (deterministic, no AI)",
             j(funnel), j({"groups": groups_out}),
             gt["id"] if gt else None, j(metrics),
             started_at.isoformat(), finished_at.isoformat(),
             int((finished_at - started_at).total_seconds()),
             0, 0, 0.0, 0),
        )
        snapshot_id = cursor.lastrowid

    return {
        "snapshot_id": snapshot_id,
        "n_groups": len(groups_out),
        "llm_calls": 0,
        "f1": (metrics or {}).get("pair_level", {}).get("f1") if metrics else None,
        "scored": metrics is not None,
    }


def run_paper_v2_analysis(series_id: int, task=None) -> dict:
    """Paper approach v2 -- the revised, higher-fidelity deterministic pipeline
    (dedup_paper_v2.py). Same four components as v1 but PAI-based product
    matching, MedDRA-coded pair screening on the event date, the real FAERS
    completeness score for reference selection, and an offline rule-based
    ETHER substitute (see dedup_paper_v2 module docstring). Entirely
    deterministic; snapshotted and scored exactly like v1."""
    started_at = datetime.utcnow()

    with connect() as conn:
        series = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if series is None:
        raise PipelineError("Series not found.")

    cases = load_series_cases(series_id)
    if not cases:
        raise PipelineError("The series has no cases. Upload a case file first.")

    class _Cancelled(Exception):
        pass

    def progress(done, total):
        if task:
            if task.total != total:
                task.set_total(total)
            task.tick()
            if task.cancelled:
                raise _Cancelled()

    try:
        result = dedup_paper_v2.run_dedup(cases, progress=progress)
    except _Cancelled:
        return {"cancelled": True}
    except RuntimeError as e:
        raise PipelineError(str(e))

    groups_out = [
        {"group_id": g["group_id"], "tag": f"paperv2_group_{g['group_id']}",
         "case_ids": sorted(set(g["case_ids"])), "reference_case": g["reference_case"]}
        for g in result["groups"]
    ]
    funnel = result["funnel"]
    finished_at = datetime.utcnow()

    with connect() as conn:
        gt = conn.execute(
            "SELECT * FROM ground_truths WHERE series_id = ? AND is_active = 1", (series_id,)
        ).fetchone()
    metrics = None
    if gt:
        universe = {str(c["safety_report_id"]) for c in cases}
        metrics = dedup_metrics.compute_metrics(
            [g["case_ids"] for g in groups_out],
            unj(gt["groups"], []),
            universe=universe,
        )

    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO snapshots(series_id, skill_version, ai_provider, model_name, funnel, result, "
            "ground_truth_id, metrics, started_at, finished_at, duration_seconds, llm_calls, total_tokens, cost, is_mock) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (series_id, PAPER_V2_SKILL_VERSION, "none",
             "InfoViP paper pipeline v2 (deterministic, no AI)",
             j(funnel), j({"groups": groups_out}),
             gt["id"] if gt else None, j(metrics),
             started_at.isoformat(), finished_at.isoformat(),
             int((finished_at - started_at).total_seconds()),
             0, 0, 0.0, 0),
        )
        snapshot_id = cursor.lastrowid

    return {
        "snapshot_id": snapshot_id,
        "n_groups": len(groups_out),
        "llm_calls": 0,
        "f1": (metrics or {}).get("pair_level", {}).get("f1") if metrics else None,
        "scored": metrics is not None,
    }


def run_paper_v2_ai_analysis(series_id: int, task=None) -> dict:
    """Enhanced Approach (AI + Narrative-Gated + Doc-Frequency Filter).
    Identical screening, scoring weights/thresholds, grouping and reference
    selection to deterministic v2; the differences -- both unique to this
    approach, absent from the published paper -- are that a small, gated
    subset of reports (see dedup_paper_v2_ai.find_gated_components) get their
    narrative features from the globally-selected AI model (ai_settings())
    instead of the rule-based ETHER substitute, and a document-frequency
    filter drops class-wide-boilerplate narrative tokens before scoring.
    Uses the same model-option setting and mock-mode toggle as the rest of
    the app."""
    started_at = datetime.utcnow()
    settings = ai_settings()
    mock = get_setting("mock_mode", False)

    with connect() as conn:
        series = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if series is None:
        raise PipelineError("Series not found.")

    cases = load_series_cases(series_id)
    if not cases:
        raise PipelineError("The series has no cases. Upload a case file first.")

    class _Cancelled(Exception):
        pass

    def progress(done, total):
        if task:
            if task.total != total:
                task.set_total(total)
            task.tick()
            if task.cancelled:
                raise _Cancelled()

    try:
        result = dedup_paper_v2_ai.run_dedup_ai(cases, settings=settings, mock=mock, progress=progress)
    except (_Cancelled, RuntimeError) as e:
        if str(e) == "Cancelled":
            return {"cancelled": True}
        raise PipelineError(str(e))

    groups_out = [
        {"group_id": g["group_id"], "tag": f"paperv2ai_group_{g['group_id']}",
         "case_ids": sorted(set(g["case_ids"])), "reference_case": g["reference_case"]}
        for g in result["groups"]
    ]
    funnel = result["funnel"]
    usage = result["usage"]
    finished_at = datetime.utcnow()

    with connect() as conn:
        gt = conn.execute(
            "SELECT * FROM ground_truths WHERE series_id = ? AND is_active = 1", (series_id,)
        ).fetchone()
    metrics = None
    if gt:
        universe = {str(c["safety_report_id"]) for c in cases}
        metrics = dedup_metrics.compute_metrics(
            [g["case_ids"] for g in groups_out],
            unj(gt["groups"], []),
            universe=universe,
        )

    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO snapshots(series_id, skill_version, ai_provider, model_name, funnel, result, "
            "ground_truth_id, metrics, started_at, finished_at, duration_seconds, llm_calls, total_tokens, cost, is_mock) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (series_id, PAPER_V2_AI_SKILL_VERSION, settings.get("provider", "none"),
             settings.get("label") or settings.get("model", "Enhanced Approach"),
             j(funnel), j({"groups": groups_out}),
             gt["id"] if gt else None, j(metrics),
             started_at.isoformat(), finished_at.isoformat(),
             int((finished_at - started_at).total_seconds()),
             funnel["ai_calls_new"], usage["input_tokens"] + usage["output_tokens"], usage["cost"],
             1 if mock else 0),
        )
        snapshot_id = cursor.lastrowid

    return {
        "snapshot_id": snapshot_id,
        "n_groups": len(groups_out),
        "llm_calls": funnel["ai_calls_new"],
        "f1": (metrics or {}).get("pair_level", {}).get("f1") if metrics else None,
        "scored": metrics is not None,
    }


def run_paper_v2_ai_literature_aware_analysis(series_id: int, task=None) -> dict:
    """Enhanced Approach + Literature-aware override -- a SEPARATE approach
    from the Enhanced Approach above (run_paper_v2_ai_analysis), which is
    intentionally left unmodified as one of the study's two main-result
    conditions. Identical pipeline, with one addition: a candidate pair is
    forced to non-duplicate whenever both reports' narratives cite the same
    journal article (dedup_paper_v2.extract_literature_source), since a
    published case-series article describes distinct patients by
    definition -- narrative overlap between two reports citing it is boiler-
    plate (shared abstract/methods text), not evidence of shared identity.
    Root-caused on SE7, the study's negative-control series; see
    documents/SE7_negative_control_analysis.md. Supportive/discussion
    evidence, not a main result."""
    started_at = datetime.utcnow()
    settings = ai_settings()
    mock = get_setting("mock_mode", False)

    with connect() as conn:
        series = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if series is None:
        raise PipelineError("Series not found.")

    cases = load_series_cases(series_id)
    if not cases:
        raise PipelineError("The series has no cases. Upload a case file first.")

    class _Cancelled(Exception):
        pass

    def progress(done, total):
        if task:
            if task.total != total:
                task.set_total(total)
            task.tick()
            if task.cancelled:
                raise _Cancelled()

    try:
        result = dedup_paper_v2_ai.run_dedup_ai(
            cases, settings=settings, mock=mock, progress=progress, literature_aware=True,
        )
    except (_Cancelled, RuntimeError) as e:
        if str(e) == "Cancelled":
            return {"cancelled": True}
        raise PipelineError(str(e))

    groups_out = [
        {"group_id": g["group_id"], "tag": f"paperv2ailit_group_{g['group_id']}",
         "case_ids": sorted(set(g["case_ids"])), "reference_case": g["reference_case"]}
        for g in result["groups"]
    ]
    funnel = result["funnel"]
    usage = result["usage"]
    finished_at = datetime.utcnow()

    with connect() as conn:
        gt = conn.execute(
            "SELECT * FROM ground_truths WHERE series_id = ? AND is_active = 1", (series_id,)
        ).fetchone()
    metrics = None
    if gt:
        universe = {str(c["safety_report_id"]) for c in cases}
        metrics = dedup_metrics.compute_metrics(
            [g["case_ids"] for g in groups_out],
            unj(gt["groups"], []),
            universe=universe,
        )

    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO snapshots(series_id, skill_version, ai_provider, model_name, funnel, result, "
            "ground_truth_id, metrics, started_at, finished_at, duration_seconds, llm_calls, total_tokens, cost, is_mock) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (series_id, PAPER_V2_AI_LIT_SKILL_VERSION, settings.get("provider", "none"),
             (settings.get("label") or settings.get("model", "Enhanced Approach")) + " (Literature-aware)",
             j(funnel), j({"groups": groups_out}),
             gt["id"] if gt else None, j(metrics),
             started_at.isoformat(), finished_at.isoformat(),
             int((finished_at - started_at).total_seconds()),
             funnel["ai_calls_new"], usage["input_tokens"] + usage["output_tokens"], usage["cost"],
             1 if mock else 0),
        )
        snapshot_id = cursor.lastrowid

    return {
        "snapshot_id": snapshot_id,
        "n_groups": len(groups_out),
        "llm_calls": funnel["ai_calls_new"],
        "f1": (metrics or {}).get("pair_level", {}).get("f1") if metrics else None,
        "scored": metrics is not None,
    }


def _run_ai_dedup(cases: list, config: dict, settings: dict, progress=None) -> dict:
    import dedup_paper
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    config_paper = config or dedup_paper.PAPER_CONFIG
    by_id = {c["case_id"]: c for c in cases}
    
    candidate_pairs = dedup_paper.screen_pairs(cases, config_paper)
    
    vocab = dedup_paper.build_drug_vocabulary(cases)
    involved = {cid for pair in candidate_pairs for cid in pair}
    features = {
        cid: dedup_paper.extract_narrative_features(by_id[cid].get("narrative"), vocab)
        for cid in involved
    }

    duplicate_pairs = []
    ambiguous_tasks = []
    
    cfg = config_paper["pairwise"]
    dup_thr = cfg["duplicate_threshold"]
    rescue_thr = cfg["rescue_threshold"]
    
    for i, (a, b) in enumerate(candidate_pairs):
        result = dedup_paper.compare_pair(by_id[a], by_id[b], features[a], features[b], config_paper)
        score = result["weight_score"]
        
        if score >= dup_thr:
            duplicate_pairs.append((a, b, score))
        elif score >= rescue_thr:
            ambiguous_tasks.append((a, b, score))
            
    # Now run AI in parallel on ambiguous tasks
    funnel = {
        "reports": len(cases),
        "candidate_pairs_screened": len(candidate_pairs),
        "ambiguous_pairs": len(ambiguous_tasks),
        "tier2_confirmed_pairs": 0,
        "llm_calls": 0,
    }
    
    if progress:
        progress(0, len(ambiguous_tasks))
        
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    lock = threading.Lock()
    cancelled = False
    
    def process_pair(a, b, score):
        nonlocal cancelled
        if cancelled: return None
        try:
            res, usage = _narrative_compare(by_id[a], by_id[b], config, settings)
        except Exception as e:
            logger.error(f"Worker narrative comparison exception: {e}")
            res = {"verdict": "uncertain", "confidence": "low"}
            usage = {}
            
        with lock:
            if cancelled: return None
            funnel["llm_calls"] += 1
            usage_total["input_tokens"] += usage.get("input_tokens", 0)
            usage_total["output_tokens"] += usage.get("output_tokens", 0)
            usage_total["cost"] += usage.get("cost", 0.0)
            
            if res.get("verdict") == "same" and res.get("confidence") in ["high", "medium"]:
                duplicate_pairs.append((a, b, score))
                funnel["tier2_confirmed_pairs"] += 1
            
        return "processed"
        
    if ambiguous_tasks:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_pair, a, b, score) for a, b, score in ambiguous_tasks]
            completed = 0
            for future in as_completed(futures):
                if progress:
                    try:
                        completed += 1
                        progress(completed, len(ambiguous_tasks))
                    except Exception:
                        cancelled = True
                res = future.result()
                if res is None:
                    cancelled = True

    if cancelled:
        raise RuntimeError("Cancelled")
        
    funnel["duplicate_pairs"] = len(duplicate_pairs)
    
    groups, graph, n_components = dedup_paper.group_and_split(duplicate_pairs)
    funnel["groups_before_split"] = n_components
    
    groups_out = []
    for i, group in enumerate(groups, start=1):
        reference = dedup_paper.select_reference_case(group, graph, by_id)
        groups_out.append({
            "group_id": i,
            "case_ids": [str(by_id[cid]["safety_report_id"]) for cid in group],
            "reference_case": str(by_id[reference]["safety_report_id"]),
        })
        
    funnel["groups_after_split"] = len(groups_out)
    return {"groups": groups_out, "funnel": funnel, "usage": usage_total}


def run_ai_analysis(series_id: int, task=None) -> dict:
    """The AI-flavored deduplication pipeline (Tiered Funnel).
    It replaces the deterministic regular expression second component with an LLM."""
    started_at = datetime.utcnow()
    from versioning import get_active_config
    version, config = get_active_config()
    settings = ai_settings()

    with connect() as conn:
        series = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if series is None:
        raise PipelineError("Series not found.")

    cases = load_series_cases(series_id)
    if not cases:
        raise PipelineError("The series has no cases. Upload a case file first.")

    class _Cancelled(Exception):
        pass

    def progress(done, total):
        if task:
            if task.total != total:
                task.set_total(total)
            task.tick()
            if task.cancelled:
                raise _Cancelled()

    try:
        result = _run_ai_dedup(cases, config, settings, progress=progress)
    except (_Cancelled, RuntimeError) as e:
        if str(e) == "Cancelled":
            return {"cancelled": True}
        raise PipelineError(str(e))

    groups_out = [
        {"group_id": g["group_id"], "tag": f"ai_group_{g['group_id']}",
         "case_ids": sorted(set(g["case_ids"])), "reference_case": g["reference_case"]}
        for g in result["groups"]
    ]
    funnel = result["funnel"]
    usage = result["usage"]
    finished_at = datetime.utcnow()

    with connect() as conn:
        gt = conn.execute(
            "SELECT * FROM ground_truths WHERE series_id = ? AND is_active = 1", (series_id,)
        ).fetchone()
    metrics = None
    if gt:
        universe = {str(c["safety_report_id"]) for c in cases}
        metrics = dedup_metrics.compute_metrics(
            [g["case_ids"] for g in groups_out],
            unj(gt["groups"], []),
            universe=universe,
        )

    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO snapshots(series_id, skill_version, ai_provider, model_name, funnel, result, "
            "ground_truth_id, metrics, started_at, finished_at, duration_seconds, llm_calls, total_tokens, cost, is_mock) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (series_id, version, settings.get("provider", "none"), settings.get("label") or settings.get("model", "InfoViP AI Pipeline"),
             j(funnel), j({"groups": groups_out}),
             gt["id"] if gt else None, j(metrics),
             started_at.isoformat(), finished_at.isoformat(),
             int((finished_at - started_at).total_seconds()),
             funnel["llm_calls"], usage["input_tokens"] + usage["output_tokens"], usage["cost"], 
             1 if get_setting("mock_mode", False) else 0),
        )
        snapshot_id = cursor.lastrowid

    return {
        "snapshot_id": snapshot_id,
        "n_groups": len(groups_out),
        "llm_calls": funnel["llm_calls"],
        "f1": (metrics or {}).get("pair_level", {}).get("f1") if metrics else None,
        "scored": metrics is not None,
    }


# --------------------------------------------------------------------------
# Batch AI audit (one call per discrepant group, concise per-case output)
# --------------------------------------------------------------------------

def _case_summary(case: dict, narrative_chars: int, comparison_tag: str = None) -> dict:
    out = {
        "safety_report_id": case["safety_report_id"],
        "age": case.get("age"), "sex": case.get("sex"), "country": case.get("country"),
        "drugs": case.get("drugs"), "events": case.get("events"),
        "initial_date": case.get("initial_date"),
        "narrative": (case.get("narrative") or "")[:narrative_chars],
    }
    if comparison_tag:
        out["comparison_tag"] = comparison_tag
    return out


def latest_scored_snapshot(series_id: int):
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM snapshots WHERE series_id = ? AND metrics IS NOT NULL ORDER BY id DESC LIMIT 1",
            (series_id,),
        ).fetchone()


def run_batch_audit(series_id: int, task=None, force: bool = False) -> dict:
    """Audits the latest scored snapshot's discrepancies group by group: one
    LLM call adjudicates all of a group's missed/new cases together. One
    concise audit row is persisted per case. Already-audited cases for the
    same snapshot are skipped unless force."""
    snap = latest_scored_snapshot(series_id)
    if snap is None:
        raise PipelineError(
            "No scored run yet. Store a ground truth, run the dedup analysis "
            "(the run is scored automatically), then start the batch audit."
        )
    metrics = unj(snap["metrics"], {})
    settings = ai_settings()
    version, config = get_active_config()
    skill_version = snap["skill_version"] if snap["skill_version"] is not None else version

    cases = load_series_cases(series_id)
    index = {str(c["safety_report_id"]): c for c in cases}
    # Each case's own bucket tag under the snapshot-relevant config, for the
    # deterministic blocking-mismatch check.
    for c in cases:
        c["computed_tag"] = dedup_core.compute_case_tag(c, config=config)

    already = set()
    if not force:
        with connect() as conn:
            rows = conn.execute(
                "SELECT safety_report_id, discrepancy_type FROM audits WHERE snapshot_id = ?", (snap["id"],)
            ).fetchall()
        already = {(r["safety_report_id"], r["discrepancy_type"]) for r in rows}

    # Reconstruct each system group's bucket tag from its members (the
    # comparison groups don't carry it).
    result_groups = unj(snap["result"], {}).get("groups") or []
    tag_by_members = {frozenset(g["case_ids"]): g.get("tag") for g in result_groups}

    n_discrepant = 0
    n_already = 0
    skipped_not_in_db = 0
    skipped_no_reference = 0
    calls = []
    for group in metrics.get("groups") or []:
        focals, matched = [], []
        for case in group.get("cases") or []:
            case_id = str(case.get("case_id"))
            tag = case.get("tag")
            rep = index.get(case_id)
            if tag in ("missed", "new"):
                n_discrepant += 1
                if (case_id, tag) in already:
                    n_already += 1
                elif rep is None:
                    skipped_not_in_db += 1
                else:
                    focals.append({**rep, "discrepancy_type": tag})
            elif tag == "matched" and rep is not None:
                matched.append(rep)
        if not focals:
            continue
        if not matched and len(focals) < 2:
            skipped_no_reference += 1
            continue
        sys_members = frozenset(
            str(c.get("case_id")) for c in (group.get("cases") or []) if c.get("tag") in ("matched", "new")
        )
        group_tag = tag_by_members.get(sys_members)
        for i in range(0, len(focals), MAX_GROUP_FOCALS):
            calls.append((group, group_tag, matched, focals[i:i + MAX_GROUP_FOCALS]))

    if task:
        task.set_total(len(calls))

    audited = 0
    failed = 0

    if calls:
        with ThreadPoolExecutor(max_workers=min(10, len(calls))) as executor:
            future_to_call = {
                executor.submit(_audit_group_call, group, group_tag, matched, focals, settings): (group, group_tag, matched, focals)
                for group, group_tag, matched, focals in calls
            }
            for future in as_completed(future_to_call):
                if task and task.cancelled:
                    break
                group, group_tag, matched, focals = future_to_call[future]
                try:
                    results = future.result()
                    saved = _persist_audits(series_id, snap["id"], skill_version, group_tag, focals, results)
                    audited += saved
                    failed += len(focals) - saved
                except Exception as e:
                    logger.error(f"Batch group audit failed (series {series_id}, tag={group_tag!r}): {e}")
                    failed += len(focals)
                if task:
                    task.tick()

    return {
        "snapshot_id": snap["id"], "skill_version": skill_version,
        "discrepant_cases": n_discrepant, "already_audited": n_already,
        "llm_calls": len(calls), "audited": audited,
        "skipped_not_in_db": skipped_not_in_db,
        "skipped_no_reference": skipped_no_reference, "failed": failed,
    }


def _audit_group_call(group: dict, group_tag: str, matched: list, focals: list, settings: dict) -> dict:
    """One LLM call adjudicating all of a group's discrepant cases. Returns
    {case_id: {verdict, confidence, likely_cause, reasoning, improvement_clue}}."""
    if get_setting("mock_mode", False):
        by_case = {}
        for f in focals:
            case_id = str(f["safety_report_id"])
            by_case[case_id] = {
                "verdict": "agree_system" if f["discrepancy_type"] == "new" else "agree_reference",
                "confidence": "high",
                "likely_cause": "blocking_cutoff" if f.get("computed_tag") != group_tag else "narrative_nuance",
                "reasoning": f"[MOCK] Simulated audit for case {case_id} ({f['discrepancy_type']}).",
                "improvement_clue": "Mock clue: adjust weights or threshold."
            }
        return by_case

    def focal_summary(f):
        out = _case_summary(f, FOCAL_NARRATIVE_CHARS)
        out["discrepancy_type"] = f["discrepancy_type"]
        out["computed_bucket_tag"] = f.get("computed_tag")
        return out

    user_message = (
        prompts.DEDUP_EVAL_GROUP_USER_PROMPT
        .replace("{group_tag}", str(group_tag or "none (ground-truth-only group)"))
        .replace("{group_status}", str(group.get("group_status") or "unknown"))
        .replace("{gt_match}", json.dumps(group.get("gt_match")) if group.get("gt_match") else "none")
        .replace("{matched_data}", json.dumps(
            [_case_summary(m, COMPANION_NARRATIVE_CHARS, "matched") for m in matched[:MAX_COMPANIONS]],
            default=str) if matched else "none")
        .replace("{discrepant_data}", json.dumps([focal_summary(f) for f in focals], default=str))
    )
    full_prompt = prompts.DEDUP_EVAL_GROUP_SYSTEM_INSTRUCTIONS + "\n\n" + user_message
    text, _usage = ai_client.generate(full_prompt, settings)
    parsed = ai_client.parse_json_response(text)

    by_case = {}
    for entry in parsed.get("cases") or []:
        if not isinstance(entry, dict) or not entry.get("case_id"):
            continue
        by_case[str(entry["case_id"])] = {
            "verdict": entry.get("verdict", "uncertain"),
            "confidence": entry.get("confidence", "low"),
            "likely_cause": entry.get("likely_cause", "other"),
            "reasoning": entry.get("reasoning", ""),
            "improvement_clue": str(entry.get("improvement_clue") or ""),
        }
    return by_case


def _persist_audits(series_id: int, snapshot_id: int, skill_version: int,
                    group_tag: str, focals: list, results: dict) -> int:
    saved = 0
    with connect() as conn:
        for focal in focals:
            case_id = str(focal["safety_report_id"])
            result = results.get(case_id)
            if result is None:
                logger.warning(f"Group audit response missing case {case_id} (series {series_id})")
                continue
            conn.execute(
                "INSERT INTO audits(series_id, snapshot_id, skill_version, safety_report_id, "
                "discrepancy_type, group_tag, computed_tag, blocking_mismatch, verdict, confidence, "
                "likely_cause, reasoning, improvement_clue) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (series_id, snapshot_id, skill_version, case_id,
                 focal["discrepancy_type"], group_tag, focal.get("computed_tag"),
                 1 if (group_tag and focal.get("computed_tag") != group_tag) else 0,
                 result["verdict"], result["confidence"], result["likely_cause"],
                 result["reasoning"], result["improvement_clue"]),
            )
            saved += 1
    return saved


# --------------------------------------------------------------------------
# Refine strategy (one LLM call over all stored audits)
# --------------------------------------------------------------------------

def _md(value) -> str:
    """Markdown-table-safe cell text."""
    return str(value if value not in (None, "") else "—").replace("|", "\\|").replace("\n", " ").strip()


def generate_audit_report(skill_version: int = None) -> str:
    """Deterministic audit report -- NO AI. Aggregates every stored audit
    (latest per case) while keeping the raw information: distributions,
    the improvement clues grouped by cause with their supporting cases, and
    the full per-case records. The corresponding skill configuration is
    embedded up front, so the report is self-contained input for the next
    refinement step (a human or an AI coding agent editing the config)."""
    from versioning import get_active_config, list_configs
    import json
    from db import unj

    with connect() as conn:
        if skill_version is not None:
            rows = conn.execute(
                "SELECT a.*, s.name AS series_name FROM audits a JOIN series s ON s.id = a.series_id "
                "WHERE a.skill_version = ? "
                "ORDER BY a.id DESC", (skill_version,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT a.*, s.name AS series_name FROM audits a JOIN series s ON s.id = a.series_id "
                "ORDER BY a.id DESC"
            ).fetchall()
    if not rows:
        if skill_version is not None:
            ver_label = _version_label(skill_version)
            raise PipelineError(f"No audits stored yet for version {ver_label}. Run batch audits first.")
        else:
            raise PipelineError("No audits stored yet. Run batch audits first.")

    # Latest audit per (series, case, type) wins.
    latest = {}
    for r in rows:
        key = (r["series_id"], r["safety_report_id"], r["discrepancy_type"])
        if key not in latest:
            latest[key] = r
    records = sorted(latest.values(), key=lambda r: (r["series_name"], r["safety_report_id"]))

    if skill_version == 0:
        display_version = 0
        display_config = dedup_paper.PAPER_CONFIG
        display_label = "Original Paper Pipeline"
    elif skill_version == PAPER_V2_SKILL_VERSION:
        display_version = PAPER_V2_SKILL_VERSION
        display_config = dedup_paper_v2.PAPER_V2_CONFIG
        display_label = "Paper Pipeline v2"
    elif skill_version == PAPER_V2_AI_SKILL_VERSION:
        display_version = PAPER_V2_AI_SKILL_VERSION
        display_config = {**dedup_paper_v2.PAPER_V2_CONFIG, "narrative_gate": dedup_paper_v2_ai.GATE_CONFIG}
        display_label = "Enhanced Approach (AI + Narrative-Gated + Doc-Frequency Filter)"
    elif skill_version is not None:
        with connect() as conn:
            cfg_row = conn.execute(
                "SELECT config, label FROM skill_configs WHERE version = ? AND is_deleted = 0",
                (skill_version,)
            ).fetchone()
        if cfg_row:
            display_version = skill_version
            display_config = unj(cfg_row["config"], {})
            display_label = cfg_row["label"]
        else:
            display_version = skill_version
            display_config = {}
            display_label = f"Version {skill_version} (Deleted)"
    else:
        active_version, active_config = get_active_config()
        active_meta = next((c for c in list_configs() if c["version"] == active_version), {})
        display_version = active_version
        display_config = active_config
        display_label = active_meta.get("label")

    counts = {
        "verdict": Counter(r["verdict"] or "?" for r in records),
        "likely_cause": Counter(r["likely_cause"] or "?" for r in records),
        "discrepancy_type": Counter(r["discrepancy_type"] or "?" for r in records),
        "skill_version": Counter(f"v{r['skill_version']}" for r in records),
        "confidence": Counter(r["confidence"] or "?" for r in records),
    }
    n_blocking = sum(1 for r in records if r["blocking_mismatch"])
    series_names = sorted({r["series_name"] for r in records})

    lines = []
    if skill_version is not None:
        ver_str = _version_label(skill_version)
        lines.append(f"# Dedup Audit Report ({ver_str})")
    else:
        lines.append("# Dedup Audit Report (All Versions)")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z · deterministic aggregation, no AI involved")
    lines.append(f"Audits: {len(records)} (latest per case) across {len(series_names)} series "
                 f"({', '.join(series_names)}) · blocking mismatches: {n_blocking}")
    lines.append("")

    # ---- the configuration being refined ---------------------------------
    lines.append(f"## Skill Configuration (v{display_version}"
                 + (f" — {display_label}" if display_label else "") + ")")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(display_config, indent=2, default=str))
    lines.append("```")
    lines.append("")

    # ---- distributions -----------------------------------------------------
    lines.append("## Aggregate Statistics")
    lines.append("")
    for title, key in [("Verdict", "verdict"), ("Likely cause", "likely_cause"),
                       ("Discrepancy type", "discrepancy_type"), ("Skill version audited", "skill_version"),
                       ("Confidence", "confidence")]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| value | count | share |")
        lines.append("|---|---:|---:|")
        for value, n in counts[key].most_common():
            lines.append(f"| {_md(value)} | {n} | {n / len(records) * 100:.1f}% |")
        lines.append("")

    # ---- raw improvement clues, grouped by cause ---------------------------
    lines.append("## Improvement Clues (raw, grouped by likely cause)")
    lines.append("")
    clues_by_cause = {}
    for r in records:
        clue = (r["improvement_clue"] or "").strip()
        if clue:
            clues_by_cause.setdefault(r["likely_cause"] or "?", {}).setdefault(clue, []).append(
                f"{r['series_name']}:{r['safety_report_id']}"
            )
    if not clues_by_cause:
        lines.append("(no improvement clues recorded)")
        lines.append("")
    for cause in sorted(clues_by_cause, key=lambda c: -sum(len(v) for v in clues_by_cause[c].values())):
        clue_map = clues_by_cause[cause]
        lines.append(f"### {cause} ({sum(len(v) for v in clue_map.values())} cases)")
        lines.append("")
        for clue, case_refs in sorted(clue_map.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"- ({len(case_refs)}×) {clue}")
            lines.append(f"  - cases: {', '.join(case_refs)}")
        lines.append("")

    # ---- full raw records ---------------------------------------------------
    lines.append("## Audit Records (raw, latest per case)")
    lines.append("")
    for series_name in series_names:
        series_records = [r for r in records if r["series_name"] == series_name]
        lines.append(f"### {series_name} ({len(series_records)} audits)")
        lines.append("")
        lines.append("| case | type | verdict | conf. | cause | group tag | case tag | blocked | reasoning | improvement clue |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in series_records:
            lines.append("| " + " | ".join([
                _md(r["safety_report_id"]), _md(r["discrepancy_type"]), _md(r["verdict"]),
                _md(r["confidence"]), _md(r["likely_cause"]), _md(r["group_tag"]),
                _md(r["computed_tag"]), "yes" if r["blocking_mismatch"] else "no",
                _md(r["reasoning"]), _md(r["improvement_clue"]),
            ]) + " |")
        lines.append("")

    return "\n".join(lines)


# Backwards-compatible alias (the endpoint used to run an LLM synthesis).
generate_refine_strategy = generate_audit_report


# --------------------------------------------------------------------------
# Overview (the monitor table: readiness + version x series metric matrix)
# --------------------------------------------------------------------------

def overview() -> dict:
    from versioning import list_configs
    configs = list_configs()

    with connect() as conn:
        series_rows = conn.execute("SELECT * FROM series ORDER BY id").fetchall()
        case_counts = dict(conn.execute(
            "SELECT series_id, COUNT(*) FROM cases GROUP BY series_id").fetchall())
        tagged_counts = dict(conn.execute(
            "SELECT series_id, COUNT(*) FROM cases WHERE tag IS NOT NULL GROUP BY series_id").fetchall())
        bucket_counts = dict(conn.execute(
            "SELECT series_id, COUNT(DISTINCT tag) FROM cases WHERE tag IS NOT NULL GROUP BY series_id").fetchall())
        gts = {r["series_id"]: r for r in conn.execute(
            "SELECT * FROM ground_truths WHERE is_active = 1").fetchall()}
        snaps = conn.execute("SELECT * FROM snapshots ORDER BY id").fetchall()
        audit_counts = dict(conn.execute(
            "SELECT series_id, COUNT(*) FROM audits GROUP BY series_id").fetchall())
        audit_by_snap = dict(conn.execute(
            "SELECT snapshot_id, COUNT(*) FROM audits WHERE snapshot_id IS NOT NULL GROUP BY snapshot_id").fetchall())
        version_audit_counts = dict(conn.execute(
            "SELECT COALESCE(skill_version, 0), COUNT(*) FROM audits GROUP BY COALESCE(skill_version, 0)"
        ).fetchall())

    snaps_by_series = {}
    for s in snaps:
        snaps_by_series.setdefault(s["series_id"], []).append(s)

    matrix_versions = set()
    out_series = []
    for s in series_rows:
        sid = s["id"]
        series_snaps = snaps_by_series.get(sid, [])
        real_snaps = [sn for sn in series_snaps if not sn["is_mock"]]
        mock_snaps = [sn for sn in series_snaps if sn["is_mock"]]
        by_version = {}
        for snap in real_snaps:
            if snap["metrics"] and snap["skill_version"] is not None:
                pair = (unj(snap["metrics"], {}) or {}).get("pair_level") or {}
                by_version[snap["skill_version"]] = {
                    "snapshot_id": snap["id"],
                    "f1": pair.get("f1"), "precision": pair.get("precision"), "recall": pair.get("recall"),
                    "model_name": snap["model_name"], "created_at": snap["created_at"],
                }
        matrix_versions.update(by_version.keys())

        series_runs = []
        for snap in series_snaps:
            pair = (unj(snap["metrics"], {}) or {}).get("pair_level") or {} if snap["metrics"] else {}
            series_runs.append({
                "id": snap["id"],
                "skill_version": snap["skill_version"],
                "model_name": snap["model_name"],
                "is_mock": bool(snap["is_mock"]),
                "created_at": snap["created_at"],
                "f1": pair.get("f1"),
                "precision": pair.get("precision"),
                "recall": pair.get("recall"),
                "n_audits": audit_by_snap.get(snap["id"], 0),
                "llm_calls": snap["llm_calls"],
                "cost": snap["cost"],
                "notes": snap["notes"],
                "is_archived": bool(snap["is_archived"]),
            })

        gt = gts.get(sid)
        out_series.append({
            "id": sid, "name": s["name"],
            "n_cases": case_counts.get(sid, 0),
            "n_tagged": tagged_counts.get(sid, 0),
            "n_buckets": bucket_counts.get(sid, 0),
            "bucketing_version": s["bucketing_version"],
            "ground_truth": {
                "stored": gt is not None,
                "n_groups": gt["n_groups"] if gt else None,
                "n_cases": gt["n_cases"] if gt else None,
                "name": gt["name"] if gt else None,
            },
            "n_runs": len(real_snaps),
            "n_mock_runs": len(mock_snaps),
            "n_scored_runs": sum(1 for snap in real_snaps if snap["metrics"]),
            "n_audits": audit_counts.get(sid, 0),
            "metrics_by_version": {str(v): cell for v, cell in by_version.items()},
            "runs": series_runs,
        })

    return {
        "skill_configs": [
            {"version": c["version"], "label": c["label"], "is_active": c["is_active"],
             "created_at": c["created_at"]}
            for c in configs
        ],
        "version_audit_counts": {str(k): v for k, v in version_audit_counts.items()},
        "series": out_series,
        "matrix_versions": sorted(matrix_versions),
    }


def latest_f1_by_series(skill_version: int, model_name: str = None) -> dict:
    """{series_id: f1} for the latest scored, non-hidden snapshot of a given
    condition (skill_version, optionally an exact model_name match).
    "Non-hidden" mirrors the frontend's isHiddenRun convention (static/
    index.html): excludes mock runs, runs flagged with a superseded/bugged
    note, and archived (retired-version) runs. Snapshots are visited in id
    order so a later row overwrites an earlier one for the same series --
    i.e. "latest" wins, same as the F1 matrix in the UI."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT series_id, model_name, metrics, is_mock, is_archived, notes "
            "FROM snapshots WHERE skill_version = ? AND metrics IS NOT NULL ORDER BY id",
            (skill_version,),
        ).fetchall()
    result = {}
    for r in rows:
        if r["is_mock"] or r["notes"] or r["is_archived"]:
            continue
        if model_name is not None and r["model_name"] != model_name:
            continue
        pair = (unj(r["metrics"], {}) or {}).get("pair_level") or {}
        f1 = pair.get("f1")
        if f1 is not None:
            result[r["series_id"]] = f1
    return result
