# Sidewalk Psych

Person B’s three data agents for studying walkability and mental distress. Built on the Wikipedia starter’s structure: agents, tools, and FastAPI routes in `app.py`; data access in `data.py`; the original context panel alongside the page. Person A owns literature, verification, and the final Writer.

## Run

```sh
uv sync
uv run app.py
```

Open [localhost:8000](http://localhost:8000). Your local `.env` is configured. For a fresh clone, copy `.env.example` to `.env` and configure your model provider. Vertex AI uses `gcloud auth application-default login` and a Google Cloud project.

To run one agent directly:

```sh
uv run app.py --agent analyst --question "Compare unadjusted and adjusted walkability–mental-distress associations."
```

## Read the code

| File | Purpose |
|---|---|
| **`app.py`** | Three agents, their tool wrappers, research runs, and web routes |
| **`data.py`** | Load the dataset, compute statistics, and attach sources/caveats to findings |
| `index.html` | Frontend |
| `context_panel.py`, `context-panel.js` | Model/tool trace panel from the starter |
| `build_data.py` | Offline source downloads, cleaning, crosswalk, and dataset export |
| `check.py` | Check the three real agents; optionally capture no-tools baselines |
| `evals.py`, `eval_questions.json` | Run the assignment’s research evaluation questions |
| `tests/` | Calculation, data-integrity, and API tests |

Start with `app.py`, then follow its calls into `data.py`. The offline build is separate from answering questions.

The built-environment agent describes EPA walkability. The health/demographics agent describes CDC and ACS measures. The analyst computes unadjusted/adjusted models and poverty interactions. All return `claim, evidence, source_ids, confidence, caveats` for the shared Writer. Numerical claims come from Python tools; the model can select only findings those tools produced.

## Data

The only shipped dataset is **`data/processed/study`**: 30,056 tracts in the 20 largest metros by Census 2023 population, with 28,150 complete adjusted-analysis cases. It joins EPA walkability, CDC modeled mental-distress estimates, and ACS demographics.

The [study notes](docs/study/README.md) explain sources, years, crosswalk weights, and quality checks. The data folder includes the main tract table, metro/county membership, coverage, crosswalk, boundary comparisons, and source/audit records.

**Coverage limitation:** CDC omits Pennsylvania’s selected health outcome; Philadelphia’s available health sample covers only its non-Pennsylvania portions. Park access, OSM amenities, direct loneliness, and social connection are not loaded.

Cross-sectional tract associations cannot establish causation or individual effects. CDC outcomes are modeled and share demographic inputs with ACS controls. The analyst reports missing cases and uses metro-clustered errors. Crosswalk estimates, source-year differences, and small cluster counts limit interpretation. No residual spatial diagnostic is computed because no adjacency data are loaded.

No invented study data are shipped. Controlled numerical inputs are generated only in pytest temporary directories to verify known answers; the app rejects synthetic snapshots.

## Validate

```sh
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run python check.py
uv run python build_data.py --check-census
```

`check.py` makes real LLM calls through `/chat`. `check.py --baselines` also makes three same-model calls with no supplied system prompt or tools. Keep API keys in `.env`.

## Rebuild data (optional)

The prepared data are included; rebuilding is only needed when changing sources or releases.

```sh
uv sync --extra build
uv run python build_data.py
```

This caches public downloads under `data/raw/study`, checks source files and geographic coverage, applies the published population crosswalk, and exports the prepared snapshot. Raw downloads are ignored by Git and Docker. The crosswalk is documented and checked in code, not independently peer reviewed.

## Questions to try

| Agent | Question | Expected result |
|---|---|---|
| Built environment | Summarize EPA walkability for the loaded 20-metro study. | Unweighted mean 11.599 across 29,485 available tracts; EPA scale is 1–20 |
| Health/demographics | Summarize mental distress, poverty, and household income. | Means: 16.011%, 11.943%, and $101,340.91, with different nonmissing sample sizes |
| Analyst | Compare unadjusted and adjusted walkability–mental-distress associations. | About +0.0563 and −0.0659 percentage points per index point; n=28,369 and 28,150, respectively; exploratory associations |

Answers come from `data/processed/study/tracts.csv` and the profile/model tools. Full questions, actual same-model no-system/no-tools answers, and agent findings are [recorded side by side](docs/study/sample-comparisons.md). Real agent traces are in [live-agent-checks.json](docs/study/live-agent-checks.json).

## Evals and team integration

```sh
uv run python evals.py --limit 3
uv run python evals.py --judge
```

Twenty hand-authored questions cover supported measures, missing layers, clarification, citations, and statistical limits. The runner has deterministic checks and an optional reference-based rubric judge. Results go to `eval_results/`; do not report scores until measured on the current data.

Person A can import `run_agent` from `app.py` and pass `result.model_dump()` to the Writer. Each call starts a fresh context. `POST /chat` accepts `message` and `agent`; responses include `response`, agent-labeled `tool_calls`, structured `result`, and `run_id`. `/run` retains the starter’s SSE activity stream. Confirm exact Project 1 fields before final team integration.

Remaining shared work: Lead fan-out/gap loop, literature/Writer/Verifier integration, combined evaluations, submission authors/URL, and Cloud Run deployment. The Dockerfile binds to `0.0.0.0:$PORT` and ships the prepared data. No deployment has been performed.
