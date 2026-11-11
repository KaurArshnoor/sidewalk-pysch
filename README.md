# Sidewalk Psych

Sidewalk Psych is a research workbench for students, planners, and public-health researchers comparing neighborhood walkability with mental distress. It combines a custom census-tract dataset for the 20 largest US metros with published research. Built on the Wikipedia starter’s structure: agents, tools, and FastAPI routes in `app.py`; data access in `data.py`; the original context panel alongside the page. Literature research, verification, and report writing integrate through the shared findings format.

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

To run a full study (planner → agents in parallel → literature–data reconciliation → Writer ↔ Verifier), use `--agent study`, or choose **Full study** on the page. The Lead starts researchers in fresh contexts in parallel, then can ask for one focused follow-up round. Each full run is saved to `runs/<run_id>.json`.

The Lead is the planner agent (`create_planner()` in `app.py`) plus `orchestrate()`. In one turn, the planner splits the question into sub-questions, one per researcher, and `orchestrate()` runs them in parallel. After that round, the planner reviews the researchers' gaps and may propose one follow-up round.

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

How to load the data:

```python
import pandas as pd

tracts = pd.read_csv("data/processed/study/tracts.csv", dtype={"geoid": str, "metro": str})
tracts[["geoid", "metro", "walkability", "distress_pct", "poverty_pct", "park_access_pct"]].describe()
```

Each row is one census tract, keyed by `geoid`, the 11-digit tract code. `metro` is the five-digit CBSA code; `metros.csv` gives the metro names. The main columns are `walkability` (EPA, 1–20), `distress_pct` (CDC PLACES frequent mental distress), the ACS measures `income`, `poverty_pct`, `black_pct`, `hispanic_pct` and `age65_pct`, plus `park_access_pct` and `amenities_per_1000`.

The [study notes](docs/study/README.md) explain sources, years, crosswalk weights, and quality checks. The data folder includes the main tract table, metro/county membership, coverage, crosswalk, boundary comparisons, and source/audit records.

**Coverage limitation:** CDC omits Pennsylvania’s selected health outcome; Philadelphia’s available health sample covers only its non-Pennsylvania portions. Park access remains missing where ParkServe does not cover every populated block point or the tract fails boundary review. Direct loneliness and social connection are not measured.

Park access is a study-derived percentage using 2020 block population points within published 10-minute walk service areas, not an official TPL tract statistic. OSM counts selected mapped features per 1,000 ACS residents; mapping omissions, duplicate representations, and fee-based venues limit interpretation. See the study notes for definitions and source terms.

Cross-sectional tract associations cannot establish causation or individual effects. CDC outcomes are modeled and share demographic inputs with ACS controls. The analyst reports missing cases and uses metro-clustered errors. Crosswalk estimates, source-year differences, and small cluster counts limit interpretation. Residual Moran’s I requires the optional tract adjacency build.

No invented study data are shipped. Controlled numerical inputs are generated only in pytest temporary directories to verify known answers; the app rejects synthetic snapshots.

## Validate

```sh
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run python -m scripts.check
uv run python -m scripts.build_data --check-census
```

`scripts/check.py` checks the live agents. Run `uv run python -m scripts.check --grader-samples` to record the three grader examples and their same-model baselines under the ignored evals/results directory. Keep API keys in `.env`.

## Rebuild data (optional)

The prepared data are included; rebuilding is only needed when changing sources or releases.

```sh
uv sync --extra build
uv run python -m scripts.build_data
```

This caches public downloads under `data/raw/study`, checks source files and geographic coverage, applies the published population crosswalk, and exports the prepared snapshot. Raw downloads are ignored by Git and Docker. To add or rebuild just the two environment layers on an existing core snapshot, run `uv run python -m scripts.build_environment`. Both builds use the same process lock and resume cached downloads. The crosswalk is documented and checked in code, not independently peer reviewed.

## Questions for the grader

Select the listed specialist and leave the metro selector at **All 20 metros**. These paired responses were captured with the same model, `vertex_ai/gemini-3.5-flash-lite`. The baseline received the identical question as a single user message.

### 1. Built environment

> Across the 20 metros, what are the mean EPA walkability score and the number of tracts with a score? Give the scale.

**Expected answer.** Mean 11.599 on EPA’s 1–20 scale, with 29,485 of 30,056 tracts scored.

**Evidence.** [tracts.csv](data/processed/study/tracts.csv), column `walkability`.

<details>
<summary>Sidewalk Psych answer</summary>

This analysis contains 30056 tracts across 20 metro areas on census2023-tract boundaries. Scope: Twenty largest US metropolitan statistical areas by Census Vintage 2023 population; full constituent counties, July 2023 OMB delineations..

EPA walkability score (1–20): mean 11.60, across 29,485 tracts; 571 missing.

</details>

<details>
<summary>Same model with no tools or system prompt</summary>

It looks like you forgot to specify which dataset or 20 metropolitan areas you are referring to!

Could you please provide the dataset, a link, or clarify which specific 20 metros and EPA walkability dataset you are analyzing? Once you share that context or the data, I would be happy to calculate the mean EPA walkability score, the number of tracts, and provide the scale for you.

</details>

### 2. Health and demographics

> Is CDC PLACES a direct survey of each tract? What period do the ACS measures cover? Answer in two sentences.

**Expected answer.** CDC PLACES produces modeled tract estimates. The ACS figures summarize the 2019–2023 five-year period.

**Evidence.** [manifest.json](data/processed/study/manifest.json), CDC PLACES and ACS source records.

<details>
<summary>Sidewalk Psych answer</summary>

CDC PLACES is modeled from BRFSS and census/ACS inputs, not a direct survey of each tract. The ACS values are period estimates from five years of survey data (2019–2023).

</details>

<details>
<summary>Same model with no tools or system prompt</summary>

No, CDC PLACES does not conduct a direct survey of every census tract; instead, it uses a small area estimation model combining CDC BRFSS survey data with Census population data to generate estimates. The American Community Survey (ACS) measures cover either 1-year periods (for populations of 65,000+) or 5-year period estimates (for all areas down to the census tract level).

</details>

### 3. Full study

> Across the 20 metros, compare unadjusted and adjusted walkability–mental-distress associations. How do they compare with published research on walkability and depression? Keep the report brief.

**Expected answer.** The pooled coefficients are about +0.0563 unadjusted (95% CI −0.0041 to 0.1167, so its direction is uncertain) and −0.0659 adjusted (95% CI −0.1004 to −0.0313), in percentage points of modeled distress per walkability point. The literature comparison depends on each paper’s measured exposure and outcome.

**Evidence.** [tracts.csv](data/processed/study/tracts.csv) via `association()` in [data.py](data.py) (the analyst’s `association` tool, unadjusted and adjusted); papers from the PubMed E-utilities and OpenAlex `/works` APIs; run through `POST /chat` or `GET /study/run`.

<details>
<summary>Sidewalk Psych answer</summary>

## What the literature says

One cross-sectional study reported that lower perceived neighborhood crime, indicative of greater neighborhood walkability, was associated with a lower odds of elevated symptoms of depression after adjusting for demographic characteristics, linguistic acculturation, and medical comorbidities [1]. One cross-sectional study reported a significant association between neighborhood walkability and depressive symptoms in men when adjusted for individual-level factors of income, physical activity, education, smoking status, living alone, age, ethnicity, and chronic disease [2]. Residence in neighbourhoods characterised by a poor quality built environment was associated with greater individual likelihood of past six month and lifetime depression in multilevel models adjusting for individual age, race/ethnicity, sex, and income and for neighbourhood level income [3].

## What our data show

In this exploratory model, the unadjusted tract-level association is 0.0563 percentage points of frequent mental distress per 1-point walkability increase, and the direction of that association is uncertain because the interval includes zero [4, 5, 6, 7, 8]. In this exploratory model, the adjusted tract-level association is -0.0659 percentage points of frequent mental distress per 1-point walkability increase [9, 4, 5, 6, 7, 8].

## Where they agree or disagree

The unadjusted tract-level estimate cannot be compared with a cross sectional study of depressive symptoms because the study measured perceived neighborhood crime, not walkability [4, 5, 6, 7, 8, 1]. Whether the unadjusted tract-level estimate and a cross sectional study of depressive symptoms agree is unclear because the data interval includes zero, so their direction is uncertain [4, 5, 6, 7, 8, 2]. The unadjusted tract-level estimate cannot be compared with a cross sectional study of past six month and lifetime depression because the study measured neighbourhood characterised by a poor quality built environment, not walkability [4, 5, 6, 7, 8, 3]. The adjusted tract-level estimate cannot be compared with a cross sectional study of depressive symptoms because the study measured perceived neighborhood crime, not walkability [9, 4, 5, 6, 7, 8, 1]. Whether the adjusted tract-level estimate and a cross sectional study of depressive symptoms agree is unclear because no single effect estimate was recorded for comparison [9, 4, 5, 6, 7, 8, 2]. The adjusted tract-level estimate cannot be compared with a cross sectional study of past six month and lifetime depression because the study measured neighbourhood characterised by a poor quality built environment, not walkability [9, 4, 5, 6, 7, 8, 3].

## Limitations and caveats

Cross-sectional tract associations cannot establish causation or individual effects [4, 5, 6, 7, 8, 9]. CDC PLACES is modeled from BRFSS and census/ACS inputs, not a direct survey of each tract [4, 5, 6, 7, 8, 9]. The study combines older built-environment measures, interpolated population weights, ACS period estimates, and modeled 2023 health outcomes [9, 4, 5, 10, 6, 7, 8, 11, 12]. In this exploratory model, residential self-selection, reverse causality, and unmeasured confounding remain possible [4, 5, 6, 7, 8, 9].

## Open gaps

- No published studies meeting criteria were found directly addressing walkability and loneliness or social connection in the retrieved search results.
- OpenAlex database search for walkability and loneliness was unavailable.
- Excluded unverifiable study: Measures of neighborhood walkability and their association with diabetes and depressive symptoms in black women (The source has no DOI or PMID to check).
- Removed an unverified statement from Question: "Across the 20 metros, compare unadjusted and adjusted walkability–mental-distress associations, and compare them with..." (worded as 'established' but the cited evidence only supports 'exploratory')
- Removed an unverified statement from Limitations and caveats: "CDC PLACES 2025 omits Pennsylvania's 2023 frequent-mental-distress estimates." (worded as 'established' but the cited evidence only supports 'exploratory')
- Removed an unverified statement from Limitations and caveats: "Cross-sectional design cannot establish direction, causation, or individual-level effects." (worded as 'established' but the cited evidence only supports 'exploratory')

## Verification

- Checked 4 paper citation(s) against Crossref/PubMed; 3 matched.
- https://openalex.org/W2527046747: not found. The source has no DOI or PMID to check
- Rounds of writing and verification: 3.
- Removed 3 sentence(s) that did not pass verification; see Open gaps.

## References

1. Rosalba Hernández, Kiarri N. Kershaw, THOMAS R. PROHASKA et al. (2014). The Cross-Sectional and Longitudinal Association Between Perceived Neighborhood Walkability Characteristics and Depressive Symptoms in Older Latinos. *Journal of Aging and Health*. https://doi.org/10.1177/0898264314553211
2. Ethan M. Berke, Laura M. Gottlieb, Anne Vernez Moudon et al. (2007). Protective Association Between Neighborhood Walkability and Depression in Older Men. *Journal of the American Geriatrics Society*. https://doi.org/10.1111/j.1532-5415.2007.01108.x
3. Sandro Galea, Jennifer Ahern, Sasha Rudenstine et al. (2005). Urban built environment and depression: a multilevel analysis. *Journal of Epidemiology & Community Health*. https://doi.org/10.1136/jech.2005.033084
4. Amos (2021), 2020 Census Block Crosswalk Data, V2. V2, CC BY 4.0; 2019 block groups → 2020 blocks. https://doi.org/10.7910/DVN/T9VMJO (retrieved 2026-11-08).
5. CDC PLACES frequent mental distress, crude prevalence. 2025 release; BRFSS MHLTH 2023. https://chronicdata.cdc.gov/resource/cwsq-ngmh.json (retrieved 2026-11-08).
6. Census TIGERweb 2023 ACS tract land areas. 2023 ACS geography, compared with Census 2020 tract inventory. https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2023/MapServer/8 (retrieved 2026-11-08).
7. Census TIGERweb ACS 2023 tract polygons. 2023 ACS tract boundaries; retrieved 2026-11-08T21:59:20.811525+00:00. https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2023/MapServer/8 (retrieved 2026-11-08).
8. EPA National Walkability Index. 2021 index; SLD 2019 geography; service GEOID20 labeled 2018; population TotPop 2018. https://edg.epa.gov/data/Public/OA/WalkabilityIndex.zip (retrieved 2026-11-08).
9. ACS 2019–2023 five-year Detailed Tables, official bulk files. 2023 ACS five-year. https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData (retrieved 2026-11-08).
10. Census 2020 populated block counts and internal points. 2020 Census; retrieved 2026-11-08T21:59:20.554324+00:00. https://tigerweb.geo.census.gov/arcgis/rest/services/Census2020/tigerWMS_Census2020/MapServer/10 (retrieved 2026-11-08).
11. OpenStreetMap contributors: mapped gathering-place features. OSM base dates 2026-11-08T21:31:10Z to 2026-11-08T21:53:35Z. https://www.openstreetmap.org/copyright (retrieved 2026-11-08).
12. Trust for Public Land ParkServe public parks and 10-minute walk service areas. Public service snapshot retrieved 2026-11-08T21:59:19.239019+00:00. https://www.tpl.org/park-data-downloads (retrieved 2026-11-08).

</details>

<details>
<summary>Same model with no tools or system prompt</summary>

### **Comparative Report: Unadjusted vs. Adjusted Walkability–Mental Distress Associations**

#### **1. Across the 20 Metros: Unadjusted vs. Adjusted Associations**
* **Unadjusted Models:** Typically show a weak-to-moderate, statistically significant inverse relationship between neighborhood walkability and mental distress. Higher walkability correlates with lower reported distress, largely because dense, walkable urban cores often provide greater access to transit, amenities, and social infrastructure.
* **Adjusted Models (controlling for confounders such as age, income, race/ethnicity, and employment):** The magnitude of the walkability effect typically **attenuates substantially**, often losing statistical significance or reversing in specific sub-populations.
* **Key Takeaway:** Socioeconomic status (SES) and individual demographics act as powerful confounders. Wealthier neighborhoods often have both high walkability and lower baseline mental distress due to financial security, healthcare access, and lower baseline stress. Once controlled, the independent protective effect of physical walkability alone is much smaller than raw data suggests.

#### **2. Comparison with Published Research**
The findings across the 20 metros align closely with the broader literature on walkability, mental health, and depression:
* **Confounding & Mediation:** Most peer-reviewed studies (e.g., Ewing et al., Beyer et al.) emphasize that neighborhood design affects mental health largely *through* intermediate mechanisms—such as physical activity, social capital, and reduced air pollution—rather than as a direct, independent cure. When models adjust for these mediators and confounders, the direct association shrinks.
* **The "Urbanicity" Paradox:** Published literature often notes mixed results in dense urban environments. While high walkability promotes mobility, hyper-dense areas can simultaneously expose residents to noise, crowding, crime, and high living costs—factors that elevate mental distress and can mask the benefits of walkability in unadjusted models.
* **Non-Linearity:** Research suggests walkability–depression relationships are rarely linear; moderate walkability combined with green space yields the most robust mental health benefits, whereas ultra-dense, car-dominated, or poorly maintained walkable areas show neutral or negative associations.

***

**Conclusion:**
Unadjusted metro data overstates the direct mental health benefits of walkability by capturing underlying socioeconomic advantages. Adjusted findings mirror published literature, confirming that while walkability supports mental well-being, its independent effect is modest once individual- and neighborhood-level socioeconomic factors are controlled.

</details>

## Evals and team integration

```sh
uv run python -m evals.run --limit 3
uv run python -m evals.run --judge
```

This project follows the **Deep Research** path. Its 25 hand-authored regression cases cover the data agents, literature search, and full studies. Deterministic checks inspect the requested models, provenance, citation screening, comparisons, and whether model caveats survived. With `--judge`, a model scores each case out of four for coverage, grounding, correct measures and scope, and uncertainty. Per-case checks, rubric judgments, findings and elapsed times are saved under `evals/results/`.

Send `{"message": "your question"}` to `POST /chat` for a full study, or set `agent` to one of the specialists. An optional five-digit `metro` code restricts the tract analysis. The response keeps `response` and `tool_calls` at the top level. Full-study `tool_calls` includes the calls from both research rounds, each with its `agent` name. The `result` and `run_id` fields include the structured report and trace reference. The page uses `/study/run` to stream progress.

Full studies screen literature citations before comparing findings. The Writer revises in response to the Verifier, and the final report retains essential limitations of the models it cites. The report, agent status, tool calls and findings are saved in `runs/<run_id>.json`. Use `--write-from` to generate a report from saved agent results.

## Cloud Run deployment

The container ships the prepared study under `data/processed/study`. It is about 19 MB in total, and each file is below GitHub’s 100 MB file limit. The Docker build excludes local credentials, raw downloads, saved runs, and eval outputs.

Deploy the `sidewalk-psych` service in the course project `ieor-test-project`. The container listens on `PORT`. Set the Cloud Run request timeout to 300 seconds. The app caps web studies at 270 seconds, with at most two research rounds.

1. Create the runtime identity `sidewalk-psych@ieor-test-project.iam.gserviceaccount.com` with the Vertex AI User role. Cloud Run uses that identity to call the configured Vertex model.
2. In the Cloud Run console, connect the GitHub repository for continuous deployment from `main`. Select the root `Dockerfile` for the build and configure the service to allow public access. Set `SIDEWALK_MODEL` to `vertex_ai/gemini-3.5-flash-lite`, `VERTEX_LOCATION` to `global`, and `GOOGLE_CLOUD_PROJECT` and `VERTEXAI_PROJECT` to `ieor-test-project`.
3. After the build deploys, open the service URL in a signed-out browser. Check `/health` and run one of the sample questions as a full study. Confirm that the report and agent-labeled tool calls arrive within five minutes.
4. Record that service URL in `submission.json` and push it to the grading branch. Confirm the course grader accounts have access to the private repository. Submit the GitHub repository URL through Courseworks.

Google documents the [GitHub continuous deployment setup](https://docs.cloud.google.com/run/docs/continuous-deployment) and [request timeout setting](https://docs.cloud.google.com/run/docs/configuring/request-timeout).
