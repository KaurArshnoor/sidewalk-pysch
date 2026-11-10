# Sidewalk Psych

Three data agents for studying walkability and mental distress. Built on the Wikipedia starter’s structure: agents, tools, and FastAPI routes in `app.py`; data access in `data.py`; the original context panel alongside the page. Literature research, verification, and report writing integrate through the shared findings format.

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

To run a full study (planner → agents in parallel → literature–data reconciliation → Writer ↔ Verifier), use `--agent study`, or choose **Full study** on the page. Each run is saved to `runs/<run_id>.json`.

```sh
uv run app.py --agent study --question "Is walkability linked to lower mental distress, and how does that compare with research on walkability and depression?"
```

## Read the code

| File | Purpose |
|---|---|
| **`app.py`** | All agents (three data agents, literature researcher, planner, Writer, Verifier), their tool wrappers, the `orchestrate()` pipeline, and web routes |
| `orchestrate.py` | Plan checks (required roles, one task per role) and literature–data reconciliation verdicts |
| **`data.py`** | Load the dataset, compute statistics, and attach sources/caveats to findings; shared findings types |
| `literature.py` | Literature search (PubMed, OpenAlex, PsyArXiv), query expansion, extraction checks, evidence-strength rubric |
| `report.py` | Writer report format, wording/citation checks, Markdown rendering |
| `verify.py` | Verifier citation checks (Crossref, PubMed, retractions), number checks, claim verdicts |
| `apis.py` | Shared HTTP access with per-service rate limits and retries |
| `web/index.html` | Frontend |
| `web/context_panel.py`, `web/context-panel.js` | Model/tool trace panel from the starter |
| `scripts/build_data.py` | Core source downloads, cleaning, crosswalk, and full dataset build |
| `scripts/build_environment.py` | ParkServe service areas, Census block points, and OSM tract counts |
| `scripts/check.py` | Check the three real agents; optionally capture no-tools baselines |
| `evals/run.py`, `evals/questions.json` | Run the assignment’s research evaluation questions |
| `tests/` | Calculation, data-integrity, and API tests |

Start with `app.py`, then follow its calls into `data.py`. Supporting code is grouped by purpose:

```text
app.py                 agents and web entry point
data.py                data tools and calculations
web/                   page and original trace panel
scripts/               offline data build and live checks
evals/                 evaluation runner and questions
tests/                 automated tests
data/processed/study/  real study dataset and audit
docs/study/            source methods and recorded results
```

The offline build is separate from answering questions.

The built-environment agent describes EPA walkability, derived ParkServe access, and mapped OSM gathering places. The health/demographics agent describes CDC and ACS measures. The analyst computes unadjusted/adjusted models and poverty or park-access interactions. All return `claim, evidence, source_ids, confidence, caveats` for the shared Writer. Numerical claims come from Python tools; the model can select only findings those tools produced.

Choose a named metro in the web form to restrict all data tools to that metro; city limits are not the unit of analysis. A selected metro shows exploratory within-metro coefficients, with confidence intervals suppressed because only one metro cluster is available. The selector is enforced in Python, rather than relying on the model to remember it. Loneliness, social connection, and explanations still require the literature findings.

## Notes on External APIs

No API keys are required, except for the Census API. The literature researcher searches PubMed, OpenAlex and PsyArXiv; the Verifier checks citations against Crossref and PubMed. The data agents and Writer call no outside services at runtime.

The main limit is **OpenAlex**: without a key, about 100 searches per day are shared by everyone on the same IP address, so it can run out on a campus network or Cloud Run. When any service is unavailable, the agent records a gap and continues. A free `OPENALEX_API_KEY` is optional. Per-service limits, fallbacks, and settings are in [docs/external-apis.md](docs/external-apis.md).

## Data

The only shipped dataset is **`data/processed/study`**: 30,056 tracts in the 20 largest metros by Census 2023 population, with 28,150 complete adjusted-analysis cases. It joins EPA walkability, CDC modeled mental-distress estimates, ACS demographics, derived ParkServe access, and OSM gathering-place counts.

The [study notes](docs/study/README.md) explain sources, years, crosswalk weights, and quality checks. The data folder includes the main tract table, metro/county membership, coverage, crosswalk, boundary comparisons, and source/audit records.

**Coverage limitation:** CDC omits Pennsylvania’s selected health outcome; Philadelphia’s available health sample covers only its non-Pennsylvania portions. Park access remains missing where ParkServe does not cover every populated block point or the tract fails boundary review. Direct loneliness and social connection are not measured.

Park access is a study-derived percentage using 2020 block population points within published 10-minute walk service areas, not an official TPL tract statistic. OSM counts selected mapped features per 1,000 ACS residents; mapping omissions, duplicate representations, and fee-based venues limit interpretation. See the study notes for definitions and source terms.

Cross-sectional tract associations cannot establish causation or individual effects. CDC outcomes are modeled and share demographic inputs with ACS controls. The analyst reports missing cases and uses metro-clustered errors. Crosswalk estimates, source-year differences, and small cluster counts limit interpretation. No residual spatial diagnostic is computed because no adjacency data are loaded.

No invented study data are shipped. Controlled numerical inputs are generated only in pytest temporary directories to verify known answers; the app rejects synthetic snapshots.

## Validate

```sh
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run python -m scripts.check
uv run python -m scripts.build_data --check-census
```

`scripts/check.py` makes real LLM calls through `/chat`. `python -m scripts.check --baselines` also makes three same-model calls with no supplied system prompt or tools. Keep API keys in `.env`.

## Rebuild data (optional)

The prepared data are included; rebuilding is only needed when changing sources or releases.

```sh
uv sync --extra build
uv run python -m scripts.build_data
```

This caches public downloads under `data/raw/study`, checks source files and geographic coverage, applies the published population crosswalk, and exports the prepared snapshot. Raw downloads are ignored by Git and Docker. To add or rebuild just the two environment layers on an existing core snapshot, run `uv run python -m scripts.build_environment`. Both builds use the same process lock and resume cached downloads. The crosswalk is documented and checked in code, not independently peer reviewed.

## Questions to try

| Agent | Question | Expected result |
|---|---|---|
| Built environment | Summarize EPA walkability for the loaded 20-metro study. | Unweighted mean 11.599 across 29,485 available tracts; EPA scale is 1–20 |
| Health/demographics | Summarize mental distress, poverty, and household income. | Means: 16.011%, 11.943%, and $101,340.91, with different nonmissing sample sizes |
| Analyst | Compare unadjusted and adjusted walkability–mental-distress associations. | About +0.0563 and −0.0659 percentage points per index point; n=28,369 and 28,150, respectively; exploratory associations |

Answers come from `data/processed/study/tracts.csv` and the profile/model tools. Full questions, actual same-model no-system/no-tools answers, and agent findings are [recorded side by side](docs/study/sample-comparisons.md). Real agent traces are in [live-agent-checks.json](docs/study/live-agent-checks.json).

## Evals and team integration

```sh
uv run python -m evals.run --limit 3
uv run python -m evals.run --judge
```

Twenty hand-authored questions cover supported measures, missing layers, clarification, citations, and statistical limits. The runner has deterministic checks and an optional reference-based rubric judge. Results go to `evals/results/`; do not report scores until measured on the current data.

The report pipeline can import `run_agent` from `app.py` and pass `result.model_dump()` to the Writer. Each call starts a fresh context. `POST /chat` accepts `message`, `agent`, and an optional `metro` code; responses include `response`, agent-labeled `tool_calls`, structured `result`, and `run_id`. `/run` retains the starter’s SSE activity stream. Confirm exact Project 1 fields before final team integration.

The literature researcher (`--agent literature`) and the Writer/Verifier loop (`write_verified_report` in `app.py`; try `--write-from` on saved results) are implemented. Remaining shared work: the Lead fan-out/gap loop that runs all agents and then the Writer/Verifier, combined evaluations, submission authors/URL, and Cloud Run deployment. The Dockerfile binds to `0.0.0.0:$PORT` and ships the prepared data. No deployment has been performed.
