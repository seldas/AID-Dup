"""
Standalone Dedup Study server -- a small Flask app, completely separate from
the main askMyFAERS application (its own SQLite database in ./data, its own
single-page interface in ./static). Focused exclusively on the dedup
research loop: upload case series -> store ground truth -> bucket -> analyze
(scored + snapshotted per skill version) -> batch AI audit -> refine
strategy -> new skill version -> repeat.

Run:  python server.py   (then open http://localhost:8555)
"""

import csv
import io
import json
import logging

from flask import Flask, jsonify, request, send_from_directory

import ai_client
import pipeline
import versioning
import dedup_paper
import dedup_paper_v2
import dedup_paper_v2_ai
import bootstrap_stats
from db import connect, init_db, j, unj, get_setting, set_setting
from pipeline import PipelineError
from tasks import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dedup_study")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# The AI-based dedup skills (bucketing + tiered LLM analysis + versioned
# skill configs) are DISABLED for the current study phase: the active
# algorithm is the deterministic paper pipeline (pipeline.run_paper_analysis,
# see documents/dedup_paper.pdf). Flip to True to re-enable the skill
# endpoints; the code paths are kept intact for later reuse.
LEGACY_SKILLS_ENABLED = True

# The v1 "(Original) Paper Approach" (run_paper_analysis) and the v1-based
# "New AI Approach (Tiered Funnel)" (run_ai_analysis) are RETIRED in favor of
# Paper Approach v2 (run_paper_v2_analysis). Their endpoints are disabled and
# hidden in the UI; the code paths are kept intact for later reuse. Flip to
# True to re-enable them (and unhide the radios in static/index.html).
V1_APPROACHES_ENABLED = False


def _require_legacy_skills():
    if not LEGACY_SKILLS_ENABLED:
        raise PipelineError(
            "The AI dedup skills are disabled in this study phase; the paper "
            "pipeline (no AI) is the active algorithm. Set LEGACY_SKILLS_ENABLED "
            "= True in server.py to re-enable them."
        )


def _require_v1_approaches():
    if not V1_APPROACHES_ENABLED:
        raise PipelineError(
            "The v1 paper approach and the v1-based AI (Tiered Funnel) approach "
            "are retired; use Paper Approach v2 instead. Set V1_APPROACHES_ENABLED "
            "= True in server.py to re-enable them."
        )


@app.errorhandler(PipelineError)
def handle_pipeline_error(e):
    return jsonify({"error": str(e)}), 409


@app.errorhandler(ai_client.AIError)
def handle_ai_error(e):
    # An unreachable/failing AI endpoint must surface as clean JSON, not a
    # Flask HTML 500 page the frontend can't parse.
    return jsonify({"error": str(e)}), 502


@app.errorhandler(ValueError)
def handle_value_error(e):
    return jsonify({"error": str(e)}), 422


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------- overview

@app.get("/api/overview")
def get_overview():
    data = pipeline.overview()
    data["active_tasks"] = registry.list(active_only=True)
    return jsonify(data)


@app.post("/api/bootstrap-compare")
def bootstrap_compare():
    """Paired bootstrap comparison of macro-F1 between two conditions across
    the benchmark series (see bootstrap_stats.py). Body:
    {"condition_a": {"skill_version": -1, "model_name": null},
     "condition_b": {"skill_version": -2, "model_name": "Claude Sonnet 4.6 (Elsa)"},
     "n_boot": 10000, "seed": 42}
    model_name is optional (omit/null for deterministic conditions that have
    no model); when given, only exact matches count. Only series with a
    scored, non-hidden run under BOTH conditions are used -- extras are
    reported back so a partial-overlap comparison is never silently
    stretched to look like a full 12-series one."""
    body = request.json or {}
    cond_a = body.get("condition_a") or {}
    cond_b = body.get("condition_b") or {}
    if "skill_version" not in cond_a or "skill_version" not in cond_b:
        return jsonify({"error": "Both condition_a and condition_b need a skill_version."}), 422
    try:
        n_boot = int(body.get("n_boot", 10000))
        seed = int(body.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify({"error": "n_boot and seed must be integers."}), 422
    if not (100 <= n_boot <= 200000):
        return jsonify({"error": "n_boot must be between 100 and 200000."}), 422

    f1_a_by_series = pipeline.latest_f1_by_series(cond_a["skill_version"], cond_a.get("model_name"))
    f1_b_by_series = pipeline.latest_f1_by_series(cond_b["skill_version"], cond_b.get("model_name"))

    common_ids = sorted(set(f1_a_by_series) & set(f1_b_by_series))
    if len(common_ids) < 2:
        return jsonify({
            "error": "Fewer than 2 series have a scored, non-hidden run under both conditions.",
            "n_condition_a": len(f1_a_by_series), "n_condition_b": len(f1_b_by_series),
        }), 422

    with connect() as conn:
        names = dict(conn.execute("SELECT id, name FROM series").fetchall())

    f1_a = [f1_a_by_series[sid] for sid in common_ids]
    f1_b = [f1_b_by_series[sid] for sid in common_ids]

    try:
        result = bootstrap_stats.bootstrap_macro_f1_comparison(f1_a, f1_b, n_boot=n_boot, seed=seed)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    result["per_series"] = [
        {"series": names.get(sid, str(sid)), "f1_a": f1_a_by_series[sid], "f1_b": f1_b_by_series[sid]}
        for sid in common_ids
    ]
    excluded_a = sorted(set(f1_a_by_series) - set(common_ids))
    excluded_b = sorted(set(f1_b_by_series) - set(common_ids))
    result["excluded_series_missing_from_a"] = [names.get(sid, str(sid)) for sid in excluded_b]
    result["excluded_series_missing_from_b"] = [names.get(sid, str(sid)) for sid in excluded_a]
    return jsonify(result)


# ------------------------------------------------------------------ series

@app.post("/api/series")
def create_series():
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "Enter a series name."}), 422
    with connect() as conn:
        existing = conn.execute("SELECT id FROM series WHERE name = ?", (name,)).fetchone()
        if existing:
            return jsonify({"error": f'A series named "{name}" already exists.'}), 422
        cursor = conn.execute("INSERT INTO series(name) VALUES(?)", (name,))
        series_id = cursor.lastrowid
    return jsonify({"id": series_id, "name": name})


@app.delete("/api/series/<int:series_id>")
def delete_series(series_id):
    with connect() as conn:
        conn.execute("DELETE FROM series WHERE id = ?", (series_id,))
    return jsonify({"message": "Series deleted (cases, ground truths, snapshots, and audits included)."})


def _get_series_or_404(series_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if row is None:
        raise PipelineError("Series not found.")
    return row


@app.post("/api/series/<int:series_id>/cases")
def upload_cases(series_id):
    _get_series_or_404(series_id)
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Attach a case file (CSV, XLSX, or JSON)."}), 422
    cases = pipeline.parse_cases_payload(file.filename, file.read())
    n = pipeline.replace_series_cases(series_id, cases)
    return jsonify({"message": f"{n} cases stored (previous cases and buckets replaced).", "n_cases": n})


@app.post("/api/series/<int:series_id>/ground-truth")
def upload_ground_truth(series_id):
    """Body: {"groups": [[case_id, ...], ...], "name"?, "source_filename"?}.
    An empty grouping is valid -- it states the series has no duplicates."""
    _get_series_or_404(series_id)
    body = request.json or {}
    raw_groups = body.get("groups")
    if not isinstance(raw_groups, list):
        return jsonify({"error": "groups must be a list of lists of case ids."}), 422
    groups = [sorted({str(c).strip() for c in g if str(c).strip()}) for g in raw_groups]
    groups = [g for g in groups if len(g) >= 2]
    n_cases = sum(len(g) for g in groups)
    with connect() as conn:
        conn.execute("UPDATE ground_truths SET is_active = 0 WHERE series_id = ? AND is_active = 1", (series_id,))
        conn.execute(
            "INSERT INTO ground_truths(series_id, name, groups, n_groups, n_cases, source_filename) "
            "VALUES(?,?,?,?,?,?)",
            (series_id,
             body.get("name") or (f"Ground truth ({len(groups)} groups)" if groups else "Ground truth (no duplicates)"),
             j(groups), len(groups), n_cases, body.get("source_filename")),
        )
    return jsonify({
        "message": "Ground truth stored.", "n_groups": len(groups), "n_cases": n_cases,
        "empty": len(groups) == 0,
    })


@app.post("/api/series/<int:series_id>/bucket")
def bucket_series(series_id):
    """LEGACY (disabled): always reset + re-run under the active skill
    version. The paper pipeline does its own pair screening instead."""
    _require_legacy_skills()
    _get_series_or_404(series_id)
    result = pipeline.run_bucketing(series_id)
    return jsonify({"message": "Bucketing complete.", **result})


@app.post("/api/series/<int:series_id>/analyze")
def analyze_series(series_id):
    """Runs the deterministic paper pipeline (Kreimeyer 2025 JBI) -- no AI
    calls, no prior bucketing needed. RETIRED: superseded by /analyze-v2."""
    _require_v1_approaches()
    series = _get_series_or_404(series_id)
    if registry.active_for_series(series_id):
        return jsonify({"error": "A task is already running for this series."}), 409
    task = registry.start(
        "ANALYZE", f"Paper dedup pipeline: {series['name']}",
        lambda t: pipeline.run_paper_analysis(series_id, task=t),
        series_id=series_id,
    )
    return jsonify({"message": "Paper pipeline started.", "task_id": task.id})


@app.post("/api/series/<int:series_id>/analyze-v2")
def analyze_series_v2(series_id):
    """Runs the revised paper approach v2 (Kreimeyer 2025 JBI, higher-fidelity
    deterministic re-implementation) -- no AI calls."""
    series = _get_series_or_404(series_id)
    if registry.active_for_series(series_id):
        return jsonify({"error": "A task is already running for this series."}), 409
    task = registry.start(
        "ANALYZE", f"Paper v2 dedup pipeline: {series['name']}",
        lambda t: pipeline.run_paper_v2_analysis(series_id, task=t),
        series_id=series_id,
    )
    return jsonify({"message": "Paper pipeline v2 started.", "task_id": task.id})


@app.post("/api/series/<int:series_id>/analyze-v2-ai")
def analyze_series_v2_ai(series_id):
    """Runs the Enhanced Approach (AI + Narrative-Gated + Doc-Frequency
    Filter): the deterministic v2 pipeline with AI-assisted narrative
    extraction swapped in only for the small, gated subset of reports where
    it's likely to matter, plus a document-frequency narrative-token filter
    -- both additions unique to this approach, absent from the published
    paper. Uses the globally selected AI model (see /api/settings) and mock
    mode."""
    series = _get_series_or_404(series_id)
    if registry.active_for_series(series_id):
        return jsonify({"error": "A task is already running for this series."}), 409
    task = registry.start(
        "ANALYZE", f"Enhanced Approach dedup pipeline: {series['name']}",
        lambda t: pipeline.run_paper_v2_ai_analysis(series_id, task=t),
        series_id=series_id,
    )
    return jsonify({"message": "Enhanced Approach pipeline started.", "task_id": task.id})


@app.post("/api/series/<int:series_id>/analyze-v2-ai-lit")
def analyze_series_v2_ai_lit(series_id):
    """Runs Enhanced Approach + Literature-aware override: identical to the
    Enhanced Approach, plus a hard rule that forces two reports to
    non-duplicate whenever their narratives cite the same journal article (a
    published case-series article describes distinct patients by
    definition). A separate, additional approach for discussion/supportive
    evidence -- the main-result Enhanced Approach above is unmodified. See
    documents/SE7_negative_control_analysis.md."""
    series = _get_series_or_404(series_id)
    if registry.active_for_series(series_id):
        return jsonify({"error": "A task is already running for this series."}), 409
    task = registry.start(
        "ANALYZE", f"Enhanced Approach (Literature-aware) dedup pipeline: {series['name']}",
        lambda t: pipeline.run_paper_v2_ai_literature_aware_analysis(series_id, task=t),
        series_id=series_id,
    )
    return jsonify({"message": "Enhanced Approach (Literature-aware) pipeline started.", "task_id": task.id})


@app.post("/api/series/<int:series_id>/analyze-ai")
def analyze_series_ai(series_id):
    """Runs the v1-based AI-augmented (Tiered Funnel) pipeline. RETIRED."""
    _require_v1_approaches()
    series = _get_series_or_404(series_id)
    if registry.active_for_series(series_id):
        return jsonify({"error": "A task is already running for this series."}), 409
    task = registry.start(
        "ANALYZE", f"AI dedup pipeline: {series['name']}",
        lambda t: pipeline.run_ai_analysis(series_id, task=t),
        series_id=series_id,
    )
    return jsonify({"message": "AI pipeline started.", "task_id": task.id})


@app.post("/api/series/<int:series_id>/analyze-legacy")
def analyze_series_legacy(series_id):
    """LEGACY (disabled): the AI-skill tiered analysis (Tier 2 / 2.5 / 3)."""
    _require_legacy_skills()
    series = _get_series_or_404(series_id)
    if registry.active_for_series(series_id):
        return jsonify({"error": "A task is already running for this series."}), 409
    task = registry.start(
        "ANALYZE", f"Dedup analysis: {series['name']}",
        lambda t: pipeline.run_analysis(series_id, task=t),
        series_id=series_id,
    )
    return jsonify({"message": "Analysis started.", "task_id": task.id})


@app.post("/api/series/<int:series_id>/batch-audit")
def batch_audit_series(series_id):
    series = _get_series_or_404(series_id)
    force = request.args.get("force", "false").lower() == "true"
    if registry.active_for_series(series_id):
        return jsonify({"error": "A task is already running for this series."}), 409
    # Fail fast instead of a FAILED background task.
    if pipeline.latest_scored_snapshot(series_id) is None:
        return jsonify({"error": "No scored run yet. Store a ground truth and run the analysis first."}), 409
    task = registry.start(
        "BATCH_AUDIT", f"Batch AI audit: {series['name']}",
        lambda t: pipeline.run_batch_audit(series_id, task=t, force=force),
        series_id=series_id,
    )
    return jsonify({"message": "Batch audit started.", "task_id": task.id})


@app.get("/api/series/<int:series_id>/snapshots")
def list_snapshots(series_id):
    _get_series_or_404(series_id)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM snapshots WHERE series_id = ? ORDER BY id DESC", (series_id,)
        ).fetchall()
    out = []
    for r in rows:
        metrics = unj(r["metrics"], {}) or {}
        out.append({
            "id": r["id"], "skill_version": r["skill_version"],
            "ai_provider": r["ai_provider"], "model_name": r["model_name"],
            "n_groups": len((unj(r["result"], {}) or {}).get("groups") or []),
            "pair_level": metrics.get("pair_level"),
            "group_level": metrics.get("group_level"),
            "funnel": unj(r["funnel"], {}),
            "duration_seconds": r["duration_seconds"], "llm_calls": r["llm_calls"],
            "total_tokens": r["total_tokens"], "cost": r["cost"],
            "is_mock": bool(r["is_mock"]),
            "created_at": r["created_at"],
            "notes": r["notes"],
            "is_archived": bool(r["is_archived"]),
        })
    return jsonify({"snapshots": out})


@app.get("/api/series/<int:series_id>/comparison")
def latest_comparison(series_id):
    """The ground-truth comparison of the latest scored snapshot (or
    ?snapshot_id=N), plus this series' stored audits keyed by case id."""
    _get_series_or_404(series_id)
    snapshot_id = request.args.get("snapshot_id", type=int)
    with connect() as conn:
        if snapshot_id:
            snap = conn.execute(
                "SELECT * FROM snapshots WHERE id = ? AND series_id = ? AND metrics IS NOT NULL",
                (snapshot_id, series_id),
            ).fetchone()
        else:
            snap = conn.execute(
                "SELECT * FROM snapshots WHERE series_id = ? AND metrics IS NOT NULL ORDER BY id DESC LIMIT 1",
                (series_id,),
            ).fetchone()
        if snap is None:
            return jsonify({"error": "No scored run snapshot for this series yet."}), 404
        audit_rows = conn.execute(
            "SELECT * FROM audits WHERE snapshot_id = ? ORDER BY id DESC", (snap["id"],)
        ).fetchall()

    audits = {}
    for a in audit_rows:  # newest first; first one per (case, type) wins
        key = f"{a['safety_report_id']}:{a['discrepancy_type']}"
        if key not in audits:
            audits[key] = {
                "verdict": a["verdict"], "confidence": a["confidence"],
                "likely_cause": a["likely_cause"], "reasoning": a["reasoning"],
                "improvement_clue": a["improvement_clue"],
                "blocking_mismatch": bool(a["blocking_mismatch"]),
                "computed_tag": a["computed_tag"], "group_tag": a["group_tag"],
            }

    metrics = unj(snap["metrics"], {}) or {}
    return jsonify({
        "snapshot_id": snap["id"], "skill_version": snap["skill_version"],
        "model_name": snap["model_name"], "created_at": snap["created_at"],
        "is_mock": bool(snap["is_mock"]),
        "duration_seconds": snap["duration_seconds"],
        "llm_calls": snap["llm_calls"],
        "total_tokens": snap["total_tokens"],
        "cost": snap["cost"],
        "notes": snap["notes"],
        "is_archived": bool(snap["is_archived"]),
        "pair_level": metrics.get("pair_level"),
        "group_level": metrics.get("group_level"),
        "gt_cases_outside_universe": metrics.get("gt_cases_outside_universe"),
        "groups": metrics.get("groups") or [],
        "audits": audits,
    })


@app.get("/api/series/<int:series_id>/audits")
def list_audits(series_id):
    _get_series_or_404(series_id)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audits WHERE series_id = ? ORDER BY id DESC", (series_id,)
        ).fetchall()
    return jsonify({"audits": [dict(r) for r in rows]})


# ----------------------------------------------------------- skill configs

@app.get("/api/skill-configs")
def get_skill_configs():
    return jsonify({"configs": versioning.list_configs()})


@app.post("/api/skill-configs")
def create_skill_config():
    _require_legacy_skills()
    body = request.json or {}
    version = versioning.create_config(
        body.get("config"), label=body.get("label"), notes=body.get("notes"),
        activate=body.get("activate", True),
    )
    return jsonify({"message": f"Skill config v{version} created.", "version": version})


@app.post("/api/skill-configs/<int:version>/activate")
def activate_skill_config(version):
    _require_legacy_skills()
    versioning.activate_config(version)
    return jsonify({"message": f"Version {version} is now active."})


@app.delete("/api/skill-configs/<int:version>")
def delete_skill_config(version):
    _require_legacy_skills()
    try:
        versioning.delete_config(version)
        return jsonify({"message": f"Version {version} deleted."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/skill-configs/<int:version>/metadata")
def update_skill_config_metadata(version):
    body = request.json or {}
    try:
        versioning.update_config_metadata(version, body.get("label"), body.get("notes"))
        return jsonify({"message": f"Version {version} metadata updated."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ------------------------------------------------------------------- tasks

@app.get("/api/tasks")
def get_tasks():
    active = request.args.get("active", "true").lower() == "true"
    return jsonify({"tasks": registry.list(active_only=active)})


@app.post("/api/tasks/<int:task_id>/cancel")
def cancel_task(task_id):
    if registry.cancel(task_id):
        return jsonify({"message": "Cancellation requested."})
    return jsonify({"error": "Task not found or not running."}), 404


# ------------------------------------------------- refine strategy / export

@app.post("/api/audit-report")
def audit_report():
    """Deterministic aggregation of all stored audits (raw information kept,
    active skill config embedded) -- NO AI call. The Markdown is also kept in
    settings for redisplay."""
    text = pipeline.generate_audit_report()
    set_setting("last_audit_report", {"text": text, "created_at": __import__("datetime").datetime.utcnow().isoformat()})
    return jsonify({"report": text})


@app.get("/api/audit-report")
def get_audit_report():
    version_str = request.args.get("version")
    if version_str is not None:
        try:
            version = int(version_str)
            text = pipeline.generate_audit_report(skill_version=version)
            return jsonify({"report": text})
        except ValueError:
            return jsonify({"error": "Invalid version value."}), 400
        except PipelineError as e:
            return jsonify({"error": str(e)}), 404
    return jsonify(get_setting("last_audit_report") or {})


@app.get("/api/export/runs.csv")
def export_runs_csv():
    """Every snapshot across all series, one row per run -- the version
    comparison table for the manuscript."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT sn.*, s.name AS series_name FROM snapshots sn JOIN series s ON s.id = sn.series_id "
            "ORDER BY sn.id"
        ).fetchall()
    header = ["snapshot_id", "series", "skill_version", "model", "n_groups",
              "precision", "recall", "f1", "tp_pairs", "fp_pairs", "fn_pairs",
              "llm_calls", "total_tokens", "cost", "duration_seconds", "created_at"]
    
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(header)
    
    for r in rows:
        metrics = unj(r["metrics"], {}) or {}
        pair = metrics.get("pair_level") or {}
        n_groups = len((unj(r["result"], {}) or {}).get("groups") or [])
        writer.writerow([
            r["id"],
            r["series_name"],
            r["skill_version"],
            r["model_name"] or "",
            n_groups,
            pair.get("precision", ""),
            pair.get("recall", ""),
            pair.get("f1", ""),
            pair.get("true_positive_pairs", ""),
            pair.get("false_positive_pairs", ""),
            pair.get("false_negative_pairs", ""),
            r["llm_calls"],
            r["total_tokens"] or "",
            r["cost"] or "",
            r["duration_seconds"],
            r["created_at"]
        ])
        
    return app.response_class(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=dedup_runs.csv"},
    )


@app.get("/api/series/<int:series_id>/cases-details")
def get_cases_details(series_id):
    _get_series_or_404(series_id)
    ids_str = request.args.get("ids", "").strip()
    if not ids_str:
        return jsonify({"cases": {}})
    ids = [i.strip() for i in ids_str.split(",") if i.strip()]
    if not ids:
        return jsonify({"cases": {}})
    with connect() as conn:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT * FROM cases WHERE series_id = ? AND safety_report_id IN ({placeholders})",
            [series_id] + ids
        ).fetchall()
    # clean_json_value: rows stored before the upload-time NaN scrub carry
    # NaN floats in `raw` (pandas float columns) -- jsonify would emit bare
    # NaN, which is invalid JSON and makes the browser drop the whole
    # response (demographics then render as em-dashes).
    out = {}
    for r in rows:
        out[r["safety_report_id"]] = pipeline.clean_json_value({
            "age": r["age"], "sex": r["sex"], "country": r["country"],
            "initial_date": r["initial_date"], "drug": r["drug"], "event": r["event"],
            "drugs": unj(r["drugs"], []), "events": unj(r["events"], []),
            "narrative": r["narrative"] or "",
            "raw": unj(r["raw"], {}),
            "ai_summary": unj(r["ai_summary"], {})
        })
    return jsonify({"cases": out})


@app.post("/api/models")
def create_or_update_model():
    body = request.json or {}
    key = body.get("key", "").strip()
    label = body.get("label", "").strip()
    provider = body.get("provider", "openai").strip()
    base_url = body.get("base_url", "").strip()
    api_key = body.get("api_key", "").strip()
    model = body.get("model", "").strip()

    if not key or not label or not base_url or not model:
        return jsonify({"error": "Fields key, label, base_url, and model are required."}), 422

    # Do not allow overwriting default hardcoded options
    if key in ["llama-3.1", "llama-4", "sonnet-4.6", "haiku-4.5"]:
         return jsonify({"error": f"Cannot modify default model key '{key}'."}), 422

    custom_models = get_setting("ai_models", {})
    custom_models[key] = {
        "label": label,
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "configured": True
    }
    set_setting("ai_models", custom_models)
    return jsonify({"message": f"Model '{label}' saved successfully."})


@app.delete("/api/models/<string:model_key>")
def delete_model(model_key):
    custom_models = get_setting("ai_models", {})
    if model_key not in custom_models:
        return jsonify({"error": f"Model '{model_key}' not found."}), 404

    # Check if this model was selected
    selected = (get_setting("ai") or {}).get("model_option")
    if selected == model_key:
        set_setting("ai", {"model_option": ai_client.DEFAULT_MODEL_OPTION})

    del custom_models[model_key]
    set_setting("ai_models", custom_models)
    return jsonify({"message": f"Model '{model_key}' deleted successfully."})


# ------------------------------------------------------------- paper config

@app.get("/api/paper-config")
def get_paper_config():
    """Exposes the active deterministic paper pipeline configuration parameters."""
    return jsonify(dedup_paper.PAPER_CONFIG)


@app.get("/api/paper-config-v2")
def get_paper_config_v2():
    """Exposes the revised paper approach v2 configuration parameters."""
    return jsonify(dedup_paper_v2.PAPER_V2_CONFIG)


@app.get("/api/paper-config-v2-ai")
def get_paper_config_v2_ai():
    """Exposes the v2 config plus the narrative-gate thresholds used to decide
    which reports get AI-assisted narrative extraction."""
    return jsonify({**dedup_paper_v2.PAPER_V2_CONFIG, "narrative_gate": dedup_paper_v2_ai.GATE_CONFIG})


# ---------------------------------------------------------------- settings

@app.get("/api/settings")
def get_ai_settings():
    """The four fixed model options (connection details resolved from .env,
    credentials never included) + which one is selected."""
    selected = (get_setting("ai") or {}).get("model_option") or ai_client.DEFAULT_MODEL_OPTION
    mock_mode = get_setting("mock_mode", False)
    return jsonify({
        "model_option": selected,
        "mock_mode": mock_mode,
        "options": [
            {"key": key, "label": o["label"], "provider": o["provider"],
             "model": o["model"], "base_url": o["base_url"], "configured": o["configured"]}
            for key, o in ai_client.model_options().items()
        ],
    })


@app.post("/api/settings")
def save_ai_settings():
    body = request.json or {}
    key = body.get("model_option")
    if key is not None:
        options = ai_client.model_options()
        if key not in options:
            return jsonify({"error": f"Unknown model option {key!r}. Choose one of: {', '.join(options)}."}), 422
        set_setting("ai", {"model_option": key})
    if "mock_mode" in body:
        set_setting("mock_mode", bool(body["mock_mode"]))
    return jsonify({"message": "Settings updated successfully."})


@app.post("/api/settings/test")
def test_ai_settings():
    try:
        settings = pipeline.ai_settings()
        text, usage = ai_client.generate("Reply with the single word: ok", settings)
        return jsonify({"message": f"{settings['label']} OK — replied: {text.strip()[:80]}", "usage": usage})
    except ai_client.AIError as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    init_db()
    print("Dedup Study running at http://localhost:8123  (database: data/dedup_study.db)")
    app.run(host="127.0.0.1", port=8123, debug=True, threaded=True)
