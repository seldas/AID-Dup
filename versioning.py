"""
Version control for the dedup skills' tunable configuration -- port of the
main app's dedup_versioning onto the standalone SQLite store. The pipeline
always runs the active version; refining the skills means creating (and
activating) a NEW version, so old versions stay runnable and comparable.
Version 1 (baseline) is auto-seeded from the code defaults on first access.
"""

import dedup_core
import prompts
from db import connect, j, unj

DEFAULT_CONFIG = {
    "screening": {
        "date_window_days": 365,
    },
    "pairwise": {
        "weights": {
            "sex":            {"match": 0.7,  "mismatch": -1.5},
            "age":            {"match": 1.5,  "mismatch": -2.0},
            "country":        {"match": 0.8,  "mismatch": -1.5},
            "date":           {"match": 1.5,  "mismatch": -1.0},
            "drugs":          {"match": 3.0,  "mismatch": -2.0},
            "events":         {"match": 2.5,  "mismatch": -2.0},
            "diagnosis":      {"match": 2.0,  "mismatch": -1.0},
            "medical_history": {"match": 1.5, "mismatch": -0.5},
            "family_history": {"match": 1.0,  "mismatch": 0.0},
            "symptoms":       {"match": 1.0,  "mismatch": -0.5},
            "narrative_drugs": {"match": 1.5, "mismatch": -0.5},
            "narrative_dates": {"match": 2.0, "mismatch": 0.0},
        },
        "age_tolerance_years": 2,
        "date_exact_bonus_days": 0,
        "date_partial_days": 30,
        "second_component_threshold": 4.5,
        "duplicate_threshold": 7.0,
        "rescue_threshold": 5.5,
        "second_component_confirm": 0.3,
    },
    "narrative_llm": {
        "merge_confidence": ["high", "medium"],
        "system_instructions": prompts.DEDUP_NARRATIVE_SYSTEM_INSTRUCTIONS,
        "user_prompt": prompts.DEDUP_NARRATIVE_USER_PROMPT,
    },
}


def validate_config(config: dict) -> list:
    """Returns human-readable problems (empty = valid)."""
    problems = []
    if not isinstance(config, dict):
        return ["config must be a JSON object"]

    screening = config.get("screening")
    if not isinstance(screening, dict):
        problems.append("screening section missing")
    else:
        if not isinstance(screening.get("date_window_days"), (int, float)) or screening.get("date_window_days", 0) <= 0:
            problems.append("screening.date_window_days must be a positive number")

    pairwise = config.get("pairwise")
    if not isinstance(pairwise, dict):
        problems.append("pairwise section missing")
    else:
        weights = pairwise.get("weights")
        if not isinstance(weights, dict):
            problems.append("pairwise.weights must be an object")
        else:
            required_keys = [
                "sex", "age", "country", "date", "drugs", "events",
                "diagnosis", "medical_history", "family_history", "symptoms",
                "narrative_drugs", "narrative_dates"
            ]
            for key in required_keys:
                w = weights.get(key)
                if not isinstance(w, dict) or "match" not in w or "mismatch" not in w:
                    problems.append(f"pairwise.weights.{key} must be an object with 'match' and 'mismatch'")
                else:
                    if not isinstance(w["match"], (int, float)):
                        problems.append(f"pairwise.weights.{key}.match must be a number")
                    if not isinstance(w["mismatch"], (int, float)):
                        problems.append(f"pairwise.weights.{key}.mismatch must be a number")

        for key in ["age_tolerance_years", "date_exact_bonus_days", "date_partial_days",
                    "second_component_threshold", "duplicate_threshold", "rescue_threshold",
                    "second_component_confirm"]:
            if not isinstance(pairwise.get(key), (int, float)):
                problems.append(f"pairwise.{key} must be a number")

    llm = config.get("narrative_llm")
    if not isinstance(llm, dict):
        problems.append("narrative_llm section missing")
    else:
        conf = llm.get("merge_confidence")
        if not isinstance(conf, list) or not conf:
            problems.append("narrative_llm.merge_confidence must be a non-empty list")
        if not isinstance(llm.get("system_instructions"), str) or not llm.get("system_instructions"):
            problems.append("narrative_llm.system_instructions must be a non-empty string")
        if not isinstance(llm.get("user_prompt"), str) or not llm.get("user_prompt"):
            problems.append("narrative_llm.user_prompt must be a non-empty string")

    return problems


def get_active_config() -> tuple:
    """(version, config_dict) of the active skill config, seeding version 1
    from the code defaults on first access."""
    with connect() as conn:
        row = conn.execute("SELECT version, config FROM skill_configs WHERE is_active = 1").fetchone()
        if row:
            return row["version"], unj(row["config"], {})
        row = conn.execute("SELECT version FROM skill_configs WHERE is_deleted = 0 ORDER BY version DESC LIMIT 1").fetchone()
        if row:
            conn.execute("UPDATE skill_configs SET is_active = 1 WHERE version = ?", (row["version"],))
            cfg = conn.execute("SELECT config FROM skill_configs WHERE version = ?", (row["version"],)).fetchone()
            return row["version"], unj(cfg["config"], {})
        conn.execute(
            "INSERT INTO skill_configs(version, label, notes, config, is_active) VALUES(1, 'baseline', ?, ?, 1)",
            ("Auto-seeded from the code defaults.", j(DEFAULT_CONFIG)),
        )
    return 1, DEFAULT_CONFIG


def list_configs() -> list:
    get_active_config()  # ensure the baseline exists
    with connect() as conn:
        rows = conn.execute("SELECT * FROM skill_configs WHERE is_deleted = 0 ORDER BY version DESC").fetchall()
    return [
        {
            "version": r["version"], "label": r["label"], "notes": r["notes"],
            "config": unj(r["config"], {}), "is_active": bool(r["is_active"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def create_config(config: dict, label: str = None, notes: str = None, activate: bool = True) -> int:
    problems = validate_config(config)
    if problems:
        raise ValueError("; ".join(problems))
    get_active_config()  # ensure the baseline exists so versions start at 1
    with connect() as conn:
        row = conn.execute("SELECT MAX(version) AS v FROM skill_configs").fetchone()
        next_version = (row["v"] or 0) + 1
        if activate:
            conn.execute("UPDATE skill_configs SET is_active = 0 WHERE is_active = 1")
        conn.execute(
            "INSERT INTO skill_configs(version, label, notes, config, is_active) VALUES(?, ?, ?, ?, ?)",
            (next_version, label, notes, j(config), 1 if activate else 0),
        )
    return next_version


def activate_config(version: int):
    with connect() as conn:
        row = conn.execute("SELECT version FROM skill_configs WHERE version = ? AND is_deleted = 0", (version,)).fetchone()
        if row is None:
            raise ValueError(f"Skill config version {version} not found")
        conn.execute("UPDATE skill_configs SET is_active = 0 WHERE is_active = 1")
        conn.execute("UPDATE skill_configs SET is_active = 1 WHERE version = ?", (version,))


def delete_config(version: int):
    with connect() as conn:
        row = conn.execute("SELECT is_active FROM skill_configs WHERE version = ? AND is_deleted = 0", (version,)).fetchone()
        if row is None:
            raise ValueError(f"Skill config version {version} not found")
        if row["is_active"]:
            raise ValueError("Cannot delete the active config. Activate another version first.")
        conn.execute("UPDATE skill_configs SET is_active = 0, is_deleted = 1 WHERE version = ?", (version,))


def update_config_metadata(version: int, label: str, notes: str):
    with connect() as conn:
        row = conn.execute("SELECT version FROM skill_configs WHERE version = ? AND is_deleted = 0", (version,)).fetchone()
        if row is None:
            raise ValueError(f"Skill config version {version} not found")
        conn.execute(
            "UPDATE skill_configs SET label = ?, notes = ? WHERE version = ?",
            (label or "", notes or "", version)
        )
