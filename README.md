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

`scripts/check.py` checks the live agents. Run `uv run python -m scripts.check --grader-samples` to record the five grader examples and their same-model baselines under the ignored evals/results directory. Keep API keys in `.env`.

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

### 3. Full study: unadjusted vs. adjusted models

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

### 4. Full study: walkability, mental health, and loneliness

> Do people in walkable neighborhoods have better mental health, less loneliness, or more social connection, and if so, why?

**Expected answer.** Partly, and only for mental health in our data. Adjusted for demographics, the tract-level association is about −0.0659 percentage points of modeled frequent mental distress per walkability point (95% CI −0.1004 to −0.0313); unadjusted it is +0.0563 with an interval that includes zero. These are neighborhood associations, not effects on individual people. The data have no loneliness or social-connection measure, so those parts come from the literature, and the comparison should mark such papers as not comparable with modeled distress. The data cannot show why; any explanation (for example physical activity or contact with neighbors) comes from published studies and should be worded as a possible mechanism, not a cause.

**Evidence.** [tracts.csv](data/processed/study/tracts.csv) via `association()` in [data.py](data.py) (the analyst’s `association` tool, unadjusted and adjusted); papers from the PubMed E-utilities, OpenAlex `/works` and PsyArXiv searches, checked against Crossref; run through `POST /chat` or `GET /study/run`.

<details>
<summary>Sidewalk Psych answer</summary>

## Question

Do people in walkable neighborhoods have better mental health, less loneliness, or more social connection, and if so, why?

## What the literature says

A longitudinal study suggests that neighborhoods with high walkability are associated with lower symptoms of hyperactivity and inattention [1]. A cross-sectional study suggests that neighborhood walkability is positively related to social interactions with neighbors [2].

## What our data show

In this exploratory model, the unadjusted tract-level association is 0.0563 percentage points of frequent mental distress per 1-point walkability increase, and the direction of that association is uncertain [3, 4, 5, 6]. In this exploratory model, the adjusted tract-level association is -0.0659 percentage points of frequent mental distress per 1-point walkability increase [7, 3, 4, 5, 6]. In this exploratory model, the adjusted tract-level association is -0.0191 percentage points of frequent mental distress per 1-point walkability increase, slope change per 10 percentage points of poverty_pct [7, 3, 4, 5, 6]. In this exploratory model, the adjusted tract-level association is 0.0003 percentage points of frequent mental distress per 1-point walkability increase, slope change per 10 percentage points of park_access_pct, and the direction of that association is uncertain [7, 3, 4, 8, 5, 6, 9].

## Where they agree or disagree

The unadjusted tract-level estimate cannot be compared with a longitudinal study of symptoms of hyperactivity and inattention because the study measured symptoms of hyperactivity and inattention, not mental distress [3, 4, 5, 6, 1]. The unadjusted tract-level estimate cannot be compared with a cross sectional study of social interactions with neighbors because the study measured social interactions with neighbors, while our data measure modeled frequent mental distress, not loneliness or social connection [3, 4, 5, 6, 2]. The adjusted tract-level estimate cannot be compared with a longitudinal study of symptoms of hyperactivity and inattention because the study measured symptoms of hyperactivity and inattention, not mental distress [7, 3, 4, 5, 6, 1]. The adjusted tract-level estimate cannot be compared with a cross sectional study of social interactions with neighbors because the study measured social interactions with neighbors, while our data measure modeled frequent mental distress, not loneliness or social connection [7, 3, 4, 5, 6, 2].

## Limitations and caveats

Cross-sectional tract associations cannot establish causation or individual effects [3, 4, 5, 6, 7, 8, 9]. CDC PLACES is modeled from BRFSS and census/ACS inputs, not a direct survey of each tract [3, 4, 5, 6, 7, 8, 9]. The study combines older built-environment measures, interpolated population weights, ACS period estimates, and modeled 2023 health outcomes [3, 4, 5, 6]. In this exploratory model, analyses and interactions are exploratory and no multiple-testing adjustment is applied [3, 4, 5, 6]. A cross-sectional study suggests that cross-sectional design cannot establish direction, causation, or individual-level effects [2].

## Open gaps

- No direct measures of perceived safety, social connection, or loneliness are included in the dataset.
- OpenStreetMap gathering-place counts represent mapped features rather than verified unique venues, and ParkServe service areas rely on study-derived block population point approximations rather than official TPL tract statistics.

## Verification

- Checked 2 paper citation(s) against Crossref/PubMed; 2 matched.
- Rounds of writing and verification: 2.

## References

1. D Parker, E Chazelas, M Melchior (2026). Neighborhood walkability with children’s mental health in the French ELFE cohort. *European Journal of Public Health*. https://doi.org/10.1093/eurpub/ckag169.274
2. Jacob Carson, Terry L. Conway, Lilian G. Perez et al. (2023). Neighborhood walkability, neighborhood social health, and self-selection among U.S. adults. *Health & Place*. https://doi.org/10.1016/j.healthplace.2023.103036
3. Amos (2021), 2020 Census Block Crosswalk Data, V2. V2, CC BY 4.0; 2019 block groups → 2020 blocks. https://doi.org/10.7910/DVN/T9VMJO (retrieved 2026-11-08).
4. CDC PLACES frequent mental distress, crude prevalence. 2025 release; BRFSS MHLTH 2023. https://chronicdata.cdc.gov/resource/cwsq-ngmh.json (retrieved 2026-11-08).
5. Census TIGERweb 2023 ACS tract land areas. 2023 ACS geography, compared with Census 2020 tract inventory. https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2023/MapServer/8 (retrieved 2026-11-08).
6. EPA National Walkability Index. 2021 index; SLD 2019 geography; service GEOID20 labeled 2018; population TotPop 2018. https://edg.epa.gov/data/Public/OA/WalkabilityIndex.zip (retrieved 2026-11-08).
7. ACS 2019–2023 five-year Detailed Tables, official bulk files. 2023 ACS five-year. https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData (retrieved 2026-11-08).
8. Census 2020 populated block counts and internal points. 2020 Census; retrieved 2026-11-08T21:59:20.554324+00:00. https://tigerweb.geo.census.gov/arcgis/rest/services/Census2020/tigerWMS_Census2020/MapServer/10 (retrieved 2026-11-08).
9. Trust for Public Land ParkServe public parks and 10-minute walk service areas. Public service snapshot retrieved 2026-11-08T21:59:19.239019+00:00. https://www.tpl.org/park-data-downloads (retrieved 2026-11-08).

</details>

<details>
<summary>Same model with no tools or system prompt</summary>

The short answer is **yes**. A vast and growing body of research in public health, urban planning, psychology, and sociology shows that people living in walkable neighborhoods generally enjoy **better mental health, lower rates of loneliness, and stronger social connections** than those living in car-dependent, sprawling areas. 

However, walkability is not a magical cure-all; rather, it acts as an **enabler** that makes positive social and psychological outcomes much easier to achieve.

Here is a breakdown of what the research says, followed by the reasons *why* walkable neighborhoods have this effect.

---

### What the Research Shows

1. **Lower Rates of Depression and Anxiety:** Studies consistently find that residents of walkable, mixed-use neighborhoods have lower risks of clinical depression and anxiety. For example, a landmark study of UK biobank data found that people who commute by walking or cycling have better mental health than those who drive.
2. **Reduced Loneliness and Social Isolation:** Walkable neighborhoods encourage "weak ties"—casual interactions with neighbors, shopkeepers, and acquaintances. Research shows these low-stakes interactions are vital for buffering against loneliness.
3. **Higher Social Capital:** Neighborhoods with pedestrian infrastructure tend to have higher "social capital," meaning residents trust each other more, feel a greater sense of community belonging, and are more likely to participate in civic life.
4. **Cognitive Benefits for Older Adults:** Walkability promotes physical activity and social engagement, both of which are strongly correlated with delayed cognitive decline and lower rates of dementia in older populations.

---

### Why Does Walkability Improve Mental Health and Connection?

Urban planners and psychologists point to several specific mechanisms that explain this connection:

#### 1. The Power of "Micro-Interactions" (The Jane Jacobs Effect)
In a car-centric suburb, you leave your garage, drive to a destination, and enter another building. You might not see a neighbor for days. 
In a walkable neighborhood, you walk out your front door to get coffee or buy groceries. Along the way, you nod at a neighbor, chat briefly with the barista, or pet a dog. Sociologists call these **"weak ties."** While they aren't deep friendships, research shows that a high frequency of weak ties significantly reduces feelings of isolation and increases a general sense of safety and belonging.

#### 2. Forced Physical Activity
Walkability inherently bakes physical activity into daily life (active transportation). Walking releases endorphins, reduces cortisol (the stress hormone), and improves sleep—all of which directly combat anxiety and depression. Furthermore, physical activity outside exposes people to natural light and green spaces, which are proven mood boosters.

#### 3. The "Eyes on the Street" Effect (Perceived Safety)
Walkable neighborhoods are denser and have more pedestrian traffic. This creates what urbanist Jane Jacobs called "eyes on the street." When public spaces feel naturally populated and lively, people feel safer. Chronic fear and hyper-vigilance—which are common in socially isolated or high-crime car-dependent areas—take a heavy toll on mental health.

#### 4. Spontaneous Socializing vs. Scheduled Socializing
In sprawling areas, seeing a friend requires planning: getting in a car, driving 20 minutes, and coordinating schedules. This high friction makes people less likely to socialize. In walkable neighborhoods, socializing is frictionless. You run into a friend at the local park or market, turning a chore into a spontaneous social event.

#### 5. Greater Autonomy for Vulnerable Populations
Children, teenagers, the elderly, and people with disabilities who cannot drive suffer immensely from the isolation of car-dependent suburbs. Walkable neighborhoods give these groups **independence**. An elderly person who can safely walk to the library or a teenager who can walk to a park is vastly less isolated than one who is entirely dependent on others for transportation.

---

### Nuance and Caveats: It’s Not Just Concrete

While the data favors walkable neighborhoods, researchers emphasize a few important caveats:

* **Gentrification and Cost:** Walkable neighborhoods are often expensive. High housing costs and gentrification can cause severe financial stress and displacement, which negate the mental health benefits of walkability. 
* **Design Quality Matters:** A walkable neighborhood with cracked sidewalks, heavy traffic noise, trash, and no trees can actually *increase* stress. Walkability must be paired with green space, aesthetic appeal, and safety to yield psychological benefits.
* **Introverts vs. Extroverts:** While walkable neighborhoods reduce loneliness on average, hyper-dense environments can sometimes overwhelm introverts or people sensitive to sensory overload if they don't have adequate access to quiet spaces.

### Summary
Walkable neighborhoods do not guarantee happiness, but they remove the architectural barriers to human connection. By turning transportation into physical activity, transforming strangers into familiar faces, and giving people the freedom to move about their communities easily, walkable design fosters the fundamental ingredients of human well-being: **movement, nature, and community.**

</details>

### 5. Full study: differences by demographics and gathering places

> How does the link between walkability and mental health, loneliness, or social connection vary with neighborhood demographics and access to public gathering places, and what evidence supports possible explanations?

**Expected answer.** The data can test two moderators. Adjusted for demographics, poverty changes the link: −0.0191 per 10 percentage points of tract poverty (95% CI −0.0342 to −0.0040), so the inverse association with distress is stronger in poorer tracts. Park access does not once demographics are controlled: +0.0003 (95% CI −0.0064 to 0.0070), although the unadjusted interaction is −0.0209 (95% CI −0.0358 to −0.0059). Gathering places can be described (a mean of 1.17 mapped features per 1,000 residents, median 0.50) but are not modeled as a moderator, and the other demographics enter only as controls. Explanations come from the literature only, and loneliness and social connection are not measured in the data.

**Evidence.** [tracts.csv](data/processed/study/tracts.csv) via `association(moderator="poverty_pct")` and `association(moderator="park_access_pct")` in [data.py](data.py), adjusted and unadjusted, plus the `amenities_per_1000` column for gathering places; papers from the PubMed E-utilities, OpenAlex `/works` and PsyArXiv searches, checked against Crossref; run through `POST /chat` or `GET /study/run`.

<details>
<summary>Sidewalk Psych answer</summary>

## Question

How does the link between walkability and mental health, loneliness, or social connection vary with neighborhood demographics and access to public gathering places, and what evidence supports possible explanations?

## What the literature says

A cross-sectional study of neighborhood walkability and social interactions with neighbors suggests that neighborhood walkability is positively related to social interactions with neighbors [1] A longitudinal study of neighborhood walkability and depressive symptoms suggests a protective association with neighborhood walkability [2] A cross-sectional study of neighborhood walkability and hyperactivity and inattention suggests that individuals living in highly walkable areas were significantly less likely to present symptoms of hyperactivity and inattention [3]

## What our data show

The unadjusted tract-level association is 0.0563 percentage points of frequent mental distress per 1-point walkability increase, and the direction of that association is uncertain [4, 5, 6, 7] The adjusted tract-level association is -0.0659 percentage points of frequent mental distress per 1-point walkability increase [8, 4, 5, 6, 7] The unadjusted tract-level association is -0.0306 percentage points of frequent mental distress per 1-point walkability increase, slope change per 10 percentage points of poverty_pct [8, 4, 5, 6, 7] The unadjusted tract-level association is -0.0209 percentage points of frequent mental distress per 1-point walkability increase, slope change per 10 percentage points of park_access_pct [4, 5, 9, 6, 7, 10]

## Where they agree or disagree

The unadjusted tract-level estimate cannot be compared with a cross sectional study of social interactions with neighbors [4, 5, 6, 7, 1] Whether the unadjusted tract-level estimate and a longitudinal study of depressive symptoms agree is unclear [4, 5, 6, 7, 2] The unadjusted tract-level estimate cannot be compared with a cross sectional study of hyperactivity and inattention [4, 5, 6, 7, 3] The adjusted tract-level estimate cannot be compared with a cross sectional study of social interactions with neighbors [8, 4, 5, 6, 7, 1] The adjusted tract-level estimate and a longitudinal study of depressive symptoms point the same way [8, 4, 5, 6, 7, 2] The adjusted tract-level estimate cannot be compared with a cross sectional study of hyperactivity and inattention [8, 4, 5, 6, 7, 3]

## Limitations and caveats

Cross-sectional tract associations cannot establish causation or individual effects [4, 5, 6, 7, 8, 9, 10]. CDC PLACES is modeled from BRFSS and census/ACS inputs, not a direct survey of each tract [4, 5, 6, 7, 8, 9, 10]. No direct loneliness or social-connection measures are loaded [4, 5, 6, 7, 8, 9, 10]. Cross-sectional design: cannot establish direction, causation, or individual-level effects [1, 3]. Measures depressive symptoms, not loneliness or social connection [2].

## Open gaps

- No direct observational measures of neighborhood social connection, actual amenity use, or perceived safety are loaded.
- Park access calculations use 2020 block population points and TPL ParkServe 10-minute walk service areas, representing a study-derived approximation rather than official tract statistics.
- Philadelphia metro distress estimates are partially omitted by CDC PLACES 2025 (Pennsylvania tracts missing).
- Published studies specifically examining the direct causal link between walkability and loneliness are sparse or inconclusive.
- Mechanisms underlying walkability, social connection, and mental health require more longitudinal and experimental evaluations.

## Verification

- Checked 3 paper citation(s) against Crossref/PubMed; 3 matched.
- Rounds of writing and verification: 1.

## References

1. Jacob Carson, Terry L. Conway, Lilian G. Perez et al. (2023). Neighborhood walkability, neighborhood social health, and self-selection among U.S. adults. *Health & Place*. https://doi.org/10.1016/j.healthplace.2023.103036
2. Ethan M. Berke, Laura M. Gottlieb, Anne Vernez Moudon et al. (2007). Protective Association Between Neighborhood Walkability and Depression in Older Men. *Journal of the American Geriatrics Society*. https://doi.org/10.1111/j.1532-5415.2007.01108.x
3. D Parker, E Chazelas, M Melchior (2026). Neighborhood walkability with children’s mental health in the French ELFE cohort. *European Journal of Public Health*. https://doi.org/10.1093/eurpub/ckag169.274
4. Amos (2021), 2020 Census Block Crosswalk Data, V2. V2, CC BY 4.0; 2019 block groups → 2020 blocks. https://doi.org/10.7910/DVN/T9VMJO (retrieved 2026-11-08).
5. CDC PLACES frequent mental distress, crude prevalence. 2025 release; BRFSS MHLTH 2023. https://chronicdata.cdc.gov/resource/cwsq-ngmh.json (retrieved 2026-11-08).
6. Census TIGERweb 2023 ACS tract land areas. 2023 ACS geography, compared with Census 2020 tract inventory. https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2023/MapServer/8 (retrieved 2026-11-08).
7. EPA National Walkability Index. 2021 index; SLD 2019 geography; service GEOID20 labeled 2018; population TotPop 2018. https://edg.epa.gov/data/Public/OA/WalkabilityIndex.zip (retrieved 2026-11-08).
8. ACS 2019–2023 five-year Detailed Tables, official bulk files. 2023 ACS five-year. https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData (retrieved 2026-11-08).
9. Census 2020 populated block counts and internal points. 2020 Census; retrieved 2026-11-08T21:59:20.554324+00:00. https://tigerweb.geo.census.gov/arcgis/rest/services/Census2020/tigerWMS_Census2020/MapServer/10 (retrieved 2026-11-08).
10. Trust for Public Land ParkServe public parks and 10-minute walk service areas. Public service snapshot retrieved 2026-11-08T21:59:19.239019+00:00. https://www.tpl.org/park-data-downloads (retrieved 2026-11-08).

</details>

<details>
<summary>Same model with no tools or system prompt</summary>

The intersection of walkability, mental health, and social connection is a rapidly growing area of public health, urban planning, and environmental psychology research. The core premise is that walkable neighborhoods reduce reliance on cars, encourage physical activity, and naturally increase “eyes on the street” and spontaneous social interactions. 

However, **this relationship is not uniform.** The mental health and social benefits of walkability vary significantly depending on neighborhood demographics (race, socioeconomic status, age) and the physical presence and quality of public gathering places. 

Here is an analysis of how these factors vary, supported by current research and underlying theoretical explanations.

---

### 1. The Role of Neighborhood Demographics

Demographic factors—particularly socioeconomic status (SES), race/ethnicity, and age—fundamentally alter how residents experience walkability and how it impacts their mental health and loneliness.

#### A. Socioeconomic Status (SES) and "The Environmental Paradox"
*   **The Discrepancy:** In high-SES neighborhoods, walkability is consistently linked to lower rates of depression, anxiety, and loneliness. However, in low-SES or historically marginalized neighborhoods, high walkability does not always yield the same mental health benefits, and sometimes correlates with *worse* outcomes (such as higher stress or perceived lack of safety).
*   **Evidence:** Studies often find that low-income, highly walkable neighborhoods frequently suffer from **"compounding environmental stressors."** While a neighborhood may have high street connectivity and dense housing, it may also feature high traffic volume, air pollution, dilapidated sidewalks, and a lack of green space. 
*   **Gentrification and Displacement:** In transitioning or gentrifying neighborhoods, increased walkability is often a precursor to rising rents and displacement. Research shows that the *fear* of displacement and the influx of unfamiliar amenities can heighten anxiety and feelings of alienation among long-term, lower-income residents, neutralizing the positive social effects of walkable infrastructure.

#### B. Race, Ethnicity, and Perceived Safety
*   **The Discrepancy:** The psychological benefit of walking relies heavily on *perceived safety*. If a neighborhood is walkable on paper (grid system, high density) but lacks social safety, the physical act of walking can induce hypervigilance rather than relaxation.
*   **Evidence:** Research in environmental justice demonstrates that minority neighborhoods are less likely to have well-maintained pedestrian infrastructure (e.g., adequate lighting, crosswalks, continuous sidewalks). Furthermore, marginalized groups may experience heightened anxiety regarding police presence, racial profiling, or street harassment while walking. Consequently, physical activity and spontaneous social interactions decrease, increasing isolation.

#### C. Age (Youth and Older Adults)
*   **Older Adults:** Walkability is a primary determinant of loneliness and mental health for seniors. In age-friendly, walkable communities, seniors experience lower rates of depression because they can independently access daily needs. Conversely, in car-centric or poorly maintained walkable areas, seniors face mobility-locked isolation.
*   **Children and Teens:** Walkability increases youth autonomy and peer socialization. However, in neighborhoods with high crime rates or heavy traffic, parents restrict children's outdoor mobility, leading to indoor isolation and increased screen time, which are linked to youth mental health challenges.

---

### 2. The Mediating Role of Public Gathering Places

Walkability alone (sidewalks and destinations) is often insufficient to combat loneliness. **Public gathering places**—often referred to as "Third Places" (cafes, parks, libraries, community centers, plazas)—act as the catalysts that turn physical movement into social connection.

*   **The Catalyst Effect:** A person can walk through a dense, highly walkable neighborhood and feel entirely isolated if there are no reasons to linger. Public spaces provide the friction necessary for human contact.
*   **Quality vs. Quantity:** It is not merely the presence of a park or square, but its *quality* and *inclusivity* that matter. High-quality public spaces feature comfortable seating, shade, programming, and maintenance. 
*   **Demographic Access to Gathering Places:** Access to well-funded, safe public spaces is deeply inequitable. Low-income and minority neighborhoods often suffer from "park deserts" or poorly maintained parks with broken amenities, which research links to higher psychological distress and community-level loneliness.

---

### 3. Possible Explanations and Supporting Evidence

Why do these variations occur? Several psychological, sociological, and physiological mechanisms explain the link between walkability, social connection, and mental health, and why they depend on demographics and public spaces:

#### A. The "Broken Windows" and Perceived Safety Hypothesis
*   **Explanation:** Physical disorder (litter, graffiti, broken sidewalks, vacant lots) signals to the brain that a neighborhood is unsafe or lacks social cohesion. This triggers a stress response (activation of the amygdala and cortisol production).
*   **Evidence:** Environmental psychology studies show that individuals walking in aesthetically pleasing, orderly environments experience greater restoration (attention restoration theory) and reductions in stress compared to those walking in visually blighted, high-walkability urban corridors. In low-SES areas, the absence of aesthetic maintenance often negates the cardiovascular and mental health benefits of walking.

#### B. Social Capital and Weak Ties Theory
*   **Explanation:** Sociologist Mark Granovetter highlighted the value of "weak ties"—casual interactions with acquaintances, neighbors, or shopkeepers. Walkability increases the frequency of these weak ties (e.g., nodding to a neighbor, chatting with a barista). Regular minor interactions significantly reduce the feeling of loneliness and foster a sense of belonging.
*   **Evidence:** Research on "Third Places" shows that frequency of visiting local gathering spots correlates directly with a higher sense of community belonging and lower loneliness. However, if public gathering places are exclusionary (e.g., high-end cafes that price out long-term residents), marginalized groups cannot form these weak ties, exacerbating social exclusion.

#### C. Environmental Stressors (Noise and Air Pollution)
*   **Explanation:** High walkability often coincides with high density and heavy traffic. Chronic exposure to traffic noise and air pollution (particulate matter) is directly linked to increased rates of depression, anxiety, and cognitive fatigue.
*   **Evidence:** Epidemiological studies controlling for physical activity find that walking along high-traffic corridors exposes individuals to pollutants that can induce systemic inflammation, which is increasingly linked by neuroscientists to depressive symptoms. Thus, the *type* of walkability (green pedestrian paths vs. congested concrete canyons) dictates the mental health outcome.

#### D. The Autonomy and Self-Determination Theory (SDT)
*   **Explanation:** Mental well-being relies on three psychological needs: autonomy, competence, and relatedness. Walkability supports autonomy (the ability to move freely without relying on others or a car) and relatedness (connecting with others).
*   **Evidence:** When older adults or low-income residents lose the ability to walk safely in their neighborhoods due to poor infrastructure or crime, their sense of autonomy plummets, directly triggering depressive symptoms and loneliness. 

---

### Summary Table: How Context Alters the Walkability-Mental Health Link

| Neighborhood Context | Impact of Walkability on Social Connection | Impact of Walkability on Mental Health | Primary Moderating Factors |
| :--- | :--- | :--- | :--- |
| **High SES / Safe / Green** | **High:** Frequent weak ties, active public spaces. | **Positive:** Lower depression/anxiety, stress reduction. | Aesthetic quality, safety, low pollution, access to green spaces. |
| **Low SES / Blighted** | **Variable/Low:** Movement without meaningful connection. | **Neutral to Negative:** Increased stress, hypervigilance. | Crime, traffic danger, physical disorder, lack of quality third places. |
| **Gentrifying** | **Friction:** Social fragmentation between old and new residents. | **Anxiety-Inducing:** Fear of displacement, alienation. | Housing cost stability, social inclusivity of new amenities. |
| **Age-Friendly (Seniors)** | **High:** Prevents isolation, maintains independence. | **Positive:** High life satisfaction, lower loneliness. | Benches, well-lit sidewalks, accessible public transport links. |

### Conclusion
Walkability is a powerful tool for improving mental health and reducing loneliness, but **it is not a neutral intervention**. Its benefits are contingent upon the socio-spatial context. To effectively leverage walkability for mental health, urban planning must move beyond mere grid density and focus on **equitable investment**—ensuring that all neighborhoods, regardless of demographics, feature safe, aesthetically pleasing pedestrian infrastructure, clean air, and inclusive public gathering places where meaningful social bonds can form.

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
