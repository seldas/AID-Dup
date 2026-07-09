"""
SQLite storage for the standalone Dedup Study app -- completely separate from
the main askMyFAERS database. Standard-library sqlite3 with JSON columns
stored as TEXT; one connection per operation (WAL mode) so background task
threads and Flask request threads don't fight over a shared handle.

Tables mirror the research subset of the main app's models:
series / cases (uploaded reports, one row per safety_report_id),
skill_configs (versioned bucketing+dedup parameters, one active),
ground_truths (per-series reference groupings, one active),
snapshots (immutable frozen run results + metrics -- the version comparison),
audits (concise per-case LLM adjudications of run-vs-GT discrepancies),
settings (key/value: AI provider config, last refine strategy).
"""

import json
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dedup_study.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    bucketing_version INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    safety_report_id TEXT NOT NULL,
    age TEXT, sex TEXT, country TEXT, initial_date TEXT,
    drug TEXT, event TEXT,
    drugs TEXT,     -- JSON list
    events TEXT,    -- JSON list
    narrative TEXT,
    raw TEXT,       -- JSON: the full uploaded row (sent to the Tier-3 LLM)
    tag TEXT,       -- bucket tag from the last bucketing run (NULL = untagged)
    ai_summary TEXT, -- JSON: extracted semantic attributes
    UNIQUE(series_id, safety_report_id)
);
CREATE INDEX IF NOT EXISTS idx_cases_series ON cases(series_id);
CREATE TABLE IF NOT EXISTS skill_configs (
    version INTEGER PRIMARY KEY,
    label TEXT, notes TEXT,
    config TEXT NOT NULL,   -- JSON
    is_active INTEGER DEFAULT 0,
    is_deleted INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS ground_truths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    name TEXT,
    groups TEXT NOT NULL,   -- JSON list of lists of safety_report_ids
    n_groups INTEGER, n_cases INTEGER,
    source_filename TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    skill_version INTEGER,
    ai_provider TEXT, model_name TEXT,
    funnel TEXT,            -- JSON tier-funnel counters
    result TEXT,            -- JSON {"groups": [{"group_id","tag","case_ids"}]}
    ground_truth_id INTEGER,
    metrics TEXT,           -- JSON from dedup_metrics.compute_metrics (NULL = unscored)
    started_at TEXT, finished_at TEXT, duration_seconds INTEGER,
    llm_calls INTEGER, total_tokens INTEGER, cost REAL,
    is_mock INTEGER DEFAULT 0,
    notes TEXT,             -- free-text annotation, e.g. flagging a superseded/buggy run
    is_archived INTEGER DEFAULT 0,  -- retired approach/version, hidden by its own toggle (not mock/bugged)
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_snapshots_series ON snapshots(series_id);
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    snapshot_id INTEGER,
    skill_version INTEGER,
    safety_report_id TEXT,
    discrepancy_type TEXT,  -- "missed" | "new"
    group_tag TEXT, computed_tag TEXT,
    blocking_mismatch INTEGER DEFAULT 0,
    verdict TEXT, confidence TEXT, likely_cause TEXT,
    reasoning TEXT, improvement_clue TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audits_series ON audits(series_id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS narrative_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    narrative_hash TEXT NOT NULL,   -- sha256 of the narrative text (content-addressed)
    model_option TEXT NOT NULL,     -- ai_client option key, e.g. "sonnet-4.6"
    prompt_version INTEGER NOT NULL,
    features TEXT NOT NULL,         -- JSON: {category: [phrase, ...], ...}
    input_tokens INTEGER, output_tokens INTEGER, cost REAL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(narrative_hash, model_option, prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_narrative_extractions_lookup
    ON narrative_extractions(narrative_hash, model_option, prompt_version);
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        # Migration: check if is_deleted column exists in skill_configs
        cursor = conn.execute("PRAGMA table_info(skill_configs)")
        cols = [r["name"] for r in cursor.fetchall()]
        if "is_deleted" not in cols:
            conn.execute("ALTER TABLE skill_configs ADD COLUMN is_deleted INTEGER DEFAULT 0")

        # Migration: check if is_mock column exists in snapshots
        cursor_snap = conn.execute("PRAGMA table_info(snapshots)")
        cols_snap = [r["name"] for r in cursor_snap.fetchall()]
        if "is_mock" not in cols_snap:
            conn.execute("ALTER TABLE snapshots ADD COLUMN is_mock INTEGER DEFAULT 0")

        # Migration: check if ai_summary column exists in cases
        cursor_cases = conn.execute("PRAGMA table_info(cases)")
        cols_cases = [r["name"] for r in cursor_cases.fetchall()]
        if "ai_summary" not in cols_cases:
            conn.execute("ALTER TABLE cases ADD COLUMN ai_summary TEXT")

        # Migration: check if notes column exists in snapshots
        if "notes" not in cols_snap:
            conn.execute("ALTER TABLE snapshots ADD COLUMN notes TEXT")

        # Migration: check if is_archived column exists in snapshots
        if "is_archived" not in cols_snap:
            conn.execute("ALTER TABLE snapshots ADD COLUMN is_archived INTEGER DEFAULT 0")


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def j(value):
    """dict/list -> JSON text for storage (None passes through)."""
    return json.dumps(value, default=str) if value is not None else None


def unj(text, default=None):
    """JSON text -> value (None/'' -> default)."""
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def get_setting(key: str, default=None):
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return unj(row["value"]) if row else default


def set_setting(key: str, value):
    with connect() as conn:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, j(value)),
        )
