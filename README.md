# Dedup Study — standalone deduplication research bench

A self-contained version of the askMyFAERS case-series deduplication research loop. **Completely separate from the main app**: its own Flask server, its own SQLite database (`data/dedup_study.db`), and its own single-page interface.

## Run

```bash
cd Dedup_Study
pip install -r requirements.txt
python server.py
# open http://localhost:8555
```

The database is created automatically on first start. Delete `data/dedup_study.db` to start over.

## Three Deduplication Approaches

The research bench supports selecting and comparing three deduplication approaches side-by-side:

1. **(Original) Paper Approach (Deterministic, no AI)**
   - Re-implementation of the literature (Kreimeyer et al., *J Biomed Inform* 165 (2025) 104824 — the FDA InfoViP FAERS deduplication pipeline), implemented in `dedup_paper.py`.
   - **No AI/LLM calls are made anywhere in this algorithm.**
   - Screening (1-year window) &rarr; Probabilistic pairwise comparison (record-linkage structured + regex features) &rarr; connected components modularity splitting &rarr; reference case selection.

2. **New AI Approach (Tiered Funnel)**
   - Extends the paper approach by replacing the rigid regex-based second component with an LLM narrative comparison.
   - Ambiguous candidate pairs falling in the log-odds *Rescue Threshold* window are adjudicated by the LLM in parallel.
   - Configured dynamically via the active **AI Config** version.

3. **Bucketing + AI Call**
   - The askMyFAERS pipeline combining pre-bucketing partition filters and AI.
   - Automatically runs **Demographic/Smart Bucketing (Tier 1)** &rarr; computes record-linkage similarity [0.0, 1.0] within buckets (Tier 2) &rarr; narrows large buckets via TF-IDF narrative pre-filtering (Tier 2.5) &rarr; adjudicates survivors via narrative LLM calls (Tier 3).
   - Also configured dynamically via the active **AI Config** version.

## Active AI Config Versioning

The settings editor (accessed via `🛠 AI Config` next to the AI radio toggle) edits the active AI settings version (e.g. `v9`, `v10`), allowing you to benchmark and compare prompts/weights side-by-side:
- **Screening Tab**: Configures candidate screening window days.
- **Log-Odds Weights Tab**: Configures matching and mismatching weights for structured fields & narrative features.
- **Thresholds & Tol Tab**: Configures log-odds thresholds (for the Tiered Funnel) and tolerances.
- **AI Prompt Tab**: Configures system instructions and user prompt templates for narrative comparisons.

## Mock AI Mode

A **Mock AI Mode** checkbox is available in the header. Toggle this on to run the entire analysis pipelines and batch audits in-memory with simulated LLM responses, bypassing all connection credential checks.

## Configure the AI models

Real LLM calls run on **four fixed model options**:

| Option | Provider | .env keys |
|---|---|---|
| Llama-3.1 | Ollama (OpenAI-compatible `/v1`) | `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL` |
| Llama-4 | vLLM (OpenAI-compatible `/v1`) | `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL` |
| Claude Sonnet 4.6 | Elsa (runPixel proxy) | `ELSA_BASE_URL`, `ELSA_API_NAME`, `ELSA_API_KEY`, `ELSA_SONNET_MODEL_ID` |
| Claude Haiku 4.5 | Elsa | same as above + `ELSA_HAIKU_MODEL_ID` |

Copy `.env.example` to `.env`, fill in the credentials, and restart the server. Open **⚙ AI Settings** in the header to select which model runs the pipeline.

## In-depth Layout

| File | Role |
|---|---|
| `server.py` | Flask API + static hosting (port 8555) |
| `pipeline.py` | ingestion, paper/AI pipeline runners, batch audits, runs snapshots, and audit reporting |
| `dedup_paper.py` | InfoViP paper pipeline re-implementation (screening, record-linkage, CNM community modularity grouping) |
| `dedup_core.py` | Skill A (bucketing) + Skill B (Tier 2/2.5/3 scoring + pre-filter) logic |
| `dedup_metrics.py` | pair/group-level scoring vs ground truth |
| `versioning.py` | versioned skill configuration schemas, defaults, and validator |
| `ai_client.py` | OpenAI-compatible and Elsa proxy client wrapper |
| `prompts.py` | narrative LLM prompts, group audit evaluation prompts |
| `tasks.py` | background task registry (monitoring, cancellation, error-reporting) |
| `db.py` | SQLite schema and database connection manager (`data/dedup_study.db`) |
| `static/index.html` | single-page research dashboard interface |
