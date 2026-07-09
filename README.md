# AID-Dup — AI Detector for Duplication

A standalone research bench for evaluating deterministic and AI-assisted duplicate-detection
strategies against expert-adjudicated FAERS pharmacovigilance case series. Self-contained: its
own Flask server, its own SQLite database (`data/dedup_study.db`), and its own single-page
interface.

This is the code accompanying the research manuscript comparing an ETHER-based deterministic
baseline, **AID-Dup** (targeted LLM narrative adjudication layered on the baseline), and an
LLM-first pipeline. The raw FAERS case dataset, ground-truth annotations, and manuscript drafts
are not included in this repository.

## Run

```bash
pip install -r requirements.txt
python server.py
# open http://localhost:8555
```

The database is created automatically on first start. Delete `data/dedup_study.db` to start over.

## Deduplication approaches

The bench supports selecting and comparing these approaches side-by-side:

1. **Paper Approach v2 (Deterministic, no AI)** — the ETHER-based baseline. A re-implementation
   of the published InfoViP FAERS deduplication algorithm (`dedup_paper_v2.py`): candidate pairs
   are screened by shared Product Active Ingredient + MedDRA adverse event within a 365-day
   window, scored with a probabilistic weighted-sum function over structured and narrative-derived
   features, linked at a calibrated duplicate threshold, and split into final groups via
   Clauset-Newman-Moore modularity community detection (with size-tiered density-exemption
   thresholds). **No AI/LLM calls are made anywhere in this algorithm.**

2. **Enhanced Approach — AID-Dup (AI + Narrative-Gated + Doc-Frequency Filter)** — preserves the
   baseline's screening, scoring, grouping, and reference-selection logic in full, and adds a
   targeted LLM narrative-adjudication step only for *narrative-pivotal* candidate pairs (pairs
   whose duplicate verdict flips depending on whether narrative-derived features are included).
   Escalation runs at the connected-component level with iterative graph rebuilding (up to 4
   rounds) and a component-size safeguard, plus a document-frequency filter that strips
   series-wide boilerplate narrative tokens (`dedup_paper_v2_ai.py`).

3. **Enhanced Approach + Literature-aware** — identical to AID-Dup, plus a hard rule against
   merging two reports that cite the same journal article. Supportive/discussion-only variant,
   not a main-result approach (see `documents/materials/SE7_negative_control_analysis.md` in the
   full research repo).

4. **Bucketing + AI Call** — the LLM-first pipeline. Does not use the baseline's scoring weights,
   duplicate threshold, or grouping algorithm. Reports are first grouped into demographic buckets
   (age band, sex, country; with fallback linkage via manufacturer-control-number/received-date
   matching), then narrowed within buckets via TF-IDF narrative pre-filtering, then adjudicated by
   direct LLM narrative comparison (`dedup_core.py`). Configured dynamically via the active **AI
   Config** version.

A retired v1 deterministic pipeline and an earlier v1-based AI approach ("Tiered Funnel") remain
in the codebase but are disabled by default (`V1_APPROACHES_ENABLED = False` in `server.py`).

## Bootstrap confidence intervals

The **📊 Bootstrap compare** tool runs a paired bootstrap (resampling with replacement across
benchmark series) over any two conditions' matched per-series macro-F1 scores, reporting point
estimates, percentile confidence intervals, and a two-sided p-value on the delta — used to
quantify whether a macro-F1 comparison is robust to which series composed the benchmark
(`bootstrap_stats.py`, `POST /api/bootstrap-compare`).

## Active AI Config versioning

The settings editor (**🛠 AI Config**, next to the AI approach radios) edits the active AI
settings version (e.g. `v9`, `v10`), allowing prompts/weights to be benchmarked and compared
side-by-side:
- **Screening Tab**: candidate screening window (days).
- **Log-Odds Weights Tab**: matching/mismatching weights for structured fields & narrative
  (ETHER) features.
- **Thresholds & Tol Tab**: log-odds thresholds and tolerances.
- **AI Prompt Tab**: system instructions and user prompt templates for narrative comparisons.

## Mock AI Mode

A **Mock AI Mode** toggle in the header runs the entire analysis pipeline and batch audits
in-memory with simulated LLM responses, bypassing all connection/credential checks — useful for
testing the app without live model access.

## Configure the AI models

Real LLM calls run on four fixed model options:

| Option | Provider | .env keys |
|---|---|---|
| Llama-3.1 | Ollama (OpenAI-compatible `/v1`) | `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL` |
| Llama-4 | vLLM (OpenAI-compatible `/v1`) | `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL` |
| Claude Sonnet 4.6 | Elsa (runPixel proxy) | `ELSA_BASE_URL`, `ELSA_API_NAME`, `ELSA_API_KEY`, `ELSA_SONNET_MODEL_ID` |
| Claude Haiku 4.5 | Elsa | same as above + `ELSA_HAIKU_MODEL_ID` |

Copy `.env.example` to `.env`, fill in the credentials, and restart the server. Open **⚙ AI
Settings** in the header to select which model runs the pipeline.

## Layout

| File | Role |
|---|---|
| `server.py` | Flask API + static hosting (port 8555) |
| `pipeline.py` | ingestion, pipeline runners, batch audits, run snapshots |
| `dedup_paper_v2.py` | ETHER-based baseline: screening, record-linkage scoring, CNM community grouping |
| `dedup_paper_v2_ai.py` | AID-Dup: narrative-pivotal gating, iterative escalation, document-frequency filter |
| `dedup_paper.py` | retired v1 baseline (kept for reference, disabled by default) |
| `dedup_core.py` | LLM-first pipeline: demographic bucketing + TF-IDF pre-filter + narrative LLM adjudication |
| `dedup_metrics.py` | pair/group-level scoring vs. ground truth |
| `bootstrap_stats.py` | paired bootstrap macro-F1 comparison (CIs, p-values) |
| `versioning.py` | versioned AI-config schemas, defaults, and validator |
| `ai_client.py` | OpenAI-compatible and Elsa proxy client wrapper |
| `prompts.py` | narrative LLM prompts, group audit evaluation prompts |
| `tasks.py` | background task registry (monitoring, cancellation, error reporting) |
| `db.py` | SQLite schema and database connection manager (`data/dedup_study.db`) |
| `static/index.html` | single-page research dashboard interface |
