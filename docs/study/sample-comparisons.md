# Real-study agent and no-tools comparisons

These comparisons were captured against the earlier EPA–CDC–ACS core, before ParkServe and OSM enrichment. They are historical examples, not evaluation of the new environment layers. Current full-snapshot agent checks are recorded in `live-agent-checks.json`.

Same model: `vertex_ai/gemini-3.5-flash-lite`. Each baseline received one user message with no supplied system prompt or tools. Baseline text is ungrounded model output, not validated research evidence. The agents read the prepared 2023-ranking core study snapshot.

## built_environment

Summarize EPA walkability for the full loaded 20-metro study. State its scale, coverage, and interpolation limits. Identify missing park-access data.

**Agent findings:**

- walkability: mean 11.599 (29485 nonmissing tracts). Sources: amos-crosswalk-v2, census-tiger-2023, epa-nwi-2021

**Coverage caveat:** Pennsylvania’s 2023 health estimates are absent from this CDC release; Philadelphia health samples represent only the available non-Pennsylvania portions.

**Same model, no system prompt or tools:**

Based on the **EPA National Walkability Index (2021 update)** methodology applied to the comprehensive, fully loaded 30-metro study (frequently expanded in spatial equity and urban form literature to represent deep metropolitan coverage), here is the summary of its scale, coverage, interpolation limits, and missing park-access data:

---

### 1. Scale and Coverage
* **Geographic Scale:** The index operates at the **Block Group** level (using U.S. Census Bureau 2010 boundaries, later aligned with 2020 geometries), providing hyper-local resolution for pedestrian accessibility.
* **Metric Scale:** It uses a relative scoring system ranging from **1 to 20**, where 1 represents the least walkable block groups and 20 represents the most walkable.
* **Metropolitan Coverage:** The full 30-metro (or fully loaded multi-metro) study covers major, diverse U.S. metropolitan statistical areas (MSAs)—balancing dense legacy cities (e.g., New York, Chicago, Boston) with sprawling auto-centric regions (e.g., Atlanta, Phoenix, Dallas-Fort Worth) to capture variance in urban design.

### 2. Interpolation Limits
* **Boundary and Edge Effects:** The underlying variables (such as intersection density and employment mix) are calculated within specific Census boundaries. At the periphery of metropolitan statistical areas, edge effects occur where adjacent non-measured rural or semi-rural areas skew density calculations.
* **Non-Continuous Variables:** The index relies on discrete network connectivity and employment data. When interpolating walkability scores between dense urban cores and exurban fringes, the model assumes smooth gradients where sharp transitions actually exist (e.g., jumping immediately from a high-density grid to a gated suburban subdivision with no through-streets).
* **Topographical and Barrier Blindness:** The mathematical interpolation of walkability scores often struggles with physical barriers (e.g., rivers, interstates, freight rail lines, steep topography) that physically sever a block group from adjacent amenities. A block group near a park or transit hub may receive a high algorithmic score, but if a highway blocks pedestrian access, the real-world interpolation breaks down.

### 3. Missing Park-Access Data
A notable limitation of the standard EPA Walkability Index methodology is its treatment of green space and parks:
* **Exclusion of Park Acreage and Quality:** The EPA Walkability Index is primarily derived from four core urban-form variables: (1) intersection density, (2) proximity to transit stops, (3) employment mix, and (4) employment/housing mix. **It does not directly measure the presence, size, or quality of parks and open spaces.**
* **Lack of Open-Space Connectivity:** While the index measures street network connectivity (intersections), it does not account for pedestrian-specific greenways, multi-use trails, or off-road paths that do not conform to the traditional street grid.
* **Implication for Equity Studies:** Because the EPA index lacks explicit park-access metrics, using it as a standalone measure of "livability" or "environmental justice" can be misleading. A neighborhood might score high on walkability due to a dense street grid and retail mix, yet suffer from a severe "park desert" lacking accessible green infrastructure.

## health_demographics

Summarize frequent mental distress, poverty, and household income in the full loaded 20-metro study. Distinguish modeled tract estimates from direct observations.

**Agent findings:**

- distress_pct: mean 16.011 (28665 nonmissing tracts); poverty_pct: mean 11.943 (29672 nonmissing tracts); income: mean 101340.911 (29459 nonmissing tracts). Sources: acs5-2023-bulk, cdc-places-2025

**Coverage caveat:** Pennsylvania’s 2023 health estimates are absent from this CDC release; Philadelphia health samples represent only the available non-Pennsylvania portions.

**Same model, no system prompt or tools:**

Based on epidemiological and public health studies analyzing geographic health disparities (such as the CDC’s 500 Cities / PLACES project or similar multi-metro health equity studies), the intersection of **frequent mental distress (FMD)**, **poverty**, and **household income** across a fully loaded 20-metro dataset reveals profound socioeconomic gradients in mental health. 

Here is a summary of these relationships, with a strict distinction between **direct observations** and **modeled tract estimates**.

---

### 1. Summary of Relationships (The Core Findings)
*   **The Income Gradient:** There is a strong, inverse relationship between household income and frequent mental distress. As median household income drops, the prevalence of FMD (defined typically as experiencing 14 or more days of poor mental health per month) rises sharply.
*   **Poverty Concentration:** Neighborhoods classified as high-poverty (often defined by the U.S. Census Bureau as tracts with poverty rates $\ge 20\%$ or severe poverty $\ge 40\%$) consistently act as hotspots for high rates of FMD. 
*   **The Urban Context across 20 Metros:** When aggregating across 20 diverse metropolitan areas, structural inequities—such as lack of green space, historical redlining, food deserts, and high housing cost burdens in low-income tracts—exacerbate mental distress independently of individual-level income.

---

### 2. Direct Observations (Empirical Data)
**Direct observations** refer to data gathered directly from individuals or physical records without statistical imputation or spatial smoothing.

*   **Source:** Typically derived from random-digit-dial telephone surveys (e.g., the Behavioral Risk Factor Surveillance System, or BRFSS) and official administrative records (e.g., U.S. Census American Community Survey [ACS] for income and poverty).
*   **What they show:** 
    *   Individual respondents living below the federal poverty line (FPL) directly report higher individual frequencies of mental distress compared to high-income respondents.
    *   Actual tax returns, wage data, and direct survey responses yield specific, unmasked data points for individuals, but **lack fine geographic granularity**. 
*   **Limitations in the Study:** Direct survey data is often too sparse at the census-tract level. If a specific low-income tract only has 15 survey respondents, the direct estimate of FMD is statistically unreliable and subject to massive margins of error. Furthermore, direct observations rarely capture the full 20-metro population uniformly due to non-response bias (e.g., highly distressed or impoverished populations are harder to reach via standard surveys).

---

### 3. Modeled Tract Estimates (Spatial & Statistical Imputation)
**Modeled tract estimates** refer to predicted values generated through multilevel regression and poststratification (MRP) or spatial smoothing models. 

*   **Source:** Created by combining direct survey observations (like BRFSS) with rich auxiliary data (like ACS demographic and economic indicators, census-tract boundaries, and spatial coordinates).
*   **What they show:** 
    *   Researchers can "fill in the gaps" for every single census tract across all 20 metropolitan areas, even those with zero direct survey respondents.
    *   Models explicitly leverage the known relationship between poverty rates, median household income, and health outcomes to *predict* the prevalence of FMD at the neighborhood (tract) level. 
    *   They visually and statistically demonstrate extreme spatial clustering: low-income tracts directly neighbor other low-income tracts, creating contiguous "zones" of elevated modeled FMD.
*   **Limitations in the Study:** Because these estimates are modeled using poverty and income as key predictors, there is a risk of **ecologic tautology** (using poverty to predict distress, which can mathematically reinforce the assumption that poor areas have high distress). Furthermore, modeled estimates represent *population averages for a tract*, meaning they may over- or under-estimate the mental distress of specific individuals living within those tracts.

### Summary of Distinction
In the 20-metro study, **direct observations** provide the ground-truth validation (actual people reporting actual income and distress), while **modeled tract estimates** provide the necessary spatial mapping to visualize how poverty and household income systematically concentrate frequent mental distress across entire metropolitan landscapes.

## analyst

Compare unadjusted and demographically adjusted walkability–frequent-mental-distress associations in the full study. Return coefficients, intervals, sample sizes, exclusions and causal/spatial caveats.

**Agent findings:**

- The unadjusted tract-level association is 0.0563 percentage points of frequent mental distress per 1-point walkability increase (n=28369; 20 metros). Sources: amos-crosswalk-v2, cdc-places-2025, census-tiger-2023, epa-nwi-2021
- The adjusted tract-level association is -0.0659 percentage points of frequent mental distress per 1-point walkability increase (n=28150; 20 metros). Sources: acs5-2023-bulk, amos-crosswalk-v2, cdc-places-2025, census-tiger-2023, epa-nwi-2021

**Coverage caveat:** Pennsylvania’s 2023 health estimates are absent from this CDC release; Philadelphia health samples represent only the available non-Pennsylvania portions.

**Same model, no system prompt or tools:**

Based on typical spatial epidemiology and environmental health studies examining the relationship between neighborhood walkability and frequent mental distress (FMD), here is a structured comparison of unadjusted and demographically adjusted models. 

*(Note: Because you did not specify the exact paper or dataset—such as a specific CDC PLACES, BRFSS, or urban health cohort study—this synthesis reflects the standard methodological framework and empirical patterns typically observed in these analyses).*

---

### 1. Model Coefficients & Confidence Intervals

In standard regression models (often multilevel logistic or linear models where FMD is measured as either a binary outcome or the number of mentally unhealthy days in the past 30):

*   **Unadjusted Model (Crude Association):**
    *   **Coefficient ($\beta$ or Odds Ratio [OR]):** Typically shows a moderately strong, statistically significant inverse relationship. For example, an OR of **0.88 to 0.92** per standard deviation increase in walkability, or a negative coefficient if using linear regression (higher walkability associated with fewer days of mental distress).
    *   **95% Confidence Interval (CI):** e.g., OR = 0.90 (95% CI: 0.87, 0.93).
    *   **Interpretation:** Without controls, people living in highly walkable neighborhoods experience lower odds of frequent mental distress.

*   **Demographically Adjusted Model:**
    *   **Coefficient ($\beta$ or OR):** The protective effect of walkability generally **attenuates (moves closer to the null/1.0)** once demographic confounders are introduced, though it frequently remains statistically significant. For example, the OR shifts to **0.94 to 0.97**.
    *   **95% Confidence Interval (CI):** e.g., OR = 0.95 (95% CI: 0.91, 0.99).
    *   **Key Demographic Adjustments Typically Included:** Age, sex, race/ethnicity, educational attainment, household income, and employment status.
    *   **Interpretation:** A substantial portion of the crude association is explained by demographic sorting (e.g., younger, higher-income, or more educated populations tend to live in more walkable urban cores and also report lower baseline mental distress). However, an independent, albeit smaller, protective association of the built environment remains.

---

### 2. Sample Sizes

*   **Full Study Sample:** Typically ranges from **$N = 10,000$ to over $100,000$** individuals, often nested within thousands of neighborhoods or census tracts (e.g., derived from state- or national-level health surveillance systems linked to GIS walkability indices like the EPA’s National Walkability Index).
*   **Sample Attrition:** The analytic sample size in the adjusted model is frequently slightly smaller than the unadjusted model due to missing covariate data (predominantly household income and educational attainment non-response).

---

### 3. Exclusions

Standard exclusion criteria in these analyses typically remove:
1.  **Missing Geographic Data:** Respondents who cannot be geocoded to a specific census block, tract, or neighborhood boundary.
2.  **Institutionalized Populations:** Individuals living in nursing homes, dormitories, or prisons where neighborhood walkability does not reflect local exposure in the same manner.
3.  **Incomplete Covariates:** Removal of observations with missing data on mental distress (the dependent variable) or key demographic/socioeconomic confounders.
4.  **Extreme Outliers:** Occasionally, rural or hyper-dense outlier tracts with zero population or anomalous built-environment features are trimmed to avoid skewing spatial weights.

---

### 4. Causal and Spatial Caveats

Interpreting the transition from unadjusted to adjusted models requires acknowledging several critical epidemiological and spatial limitations:

*   **Residential Self-Selection (Causal Caveat):** 
    *   Cross-sectional studies struggle to prove causation. People with a preference for active, walkable lifestyles and better baseline mental health actively choose to move to walkable neighborhoods. Thus, the remaining association in adjusted models may still partly reflect underlying psychological traits rather than a direct causal impact of the built environment.
*   **Confounding by Indication / Socioeconomic Status (SES):** 
    *   While demographic adjustments (income, education) help, residual confounding remains. Walkable urban areas are frequently gentrified, meaning poverty and neighborhood deprivation are spatially intertwined with walkability in complex ways.
*   **Spatial Autocorrelation and Modifiable Areal Unit Problem (MAUP):** 
    *   Health outcomes and environmental exposures are spatially dependent (people living near one another share unmeasured environmental and social characteristics). Standard regression standard errors may be artificially deflated unless spatial regression, cluster-robust standard errors, or multilevel models are employed. Furthermore, the definition of the "neighborhood" boundary (census tract vs. block group) can alter the magnitude of the walkability coefficient.
*   **The "Dark Side" of Walkability:** 
    *   High walkability can also correlate with stressors such as traffic noise, air pollution, crowding, and crime, which may act as suppressors or mediators that complicate the net mental health effect if not fully modeled.

***

*If you are referring to a specific published paper, please provide the authors or title, and these figures can be updated with the exact reported statistics.*

