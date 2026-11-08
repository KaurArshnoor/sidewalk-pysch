# 20-metro dataset — build and crosswalk review

Built from public sources: **30,056 tracts, 20 metropolitan areas, 186 counties**. The primary adjusted model has **28,150 complete cases** after source and boundary quality checks. The EPA–CDC–ACS core is supplemented with derived ParkServe access and mapped OSM gathering places. Direct loneliness and social connection are not measured.

## ParkServe and OpenStreetMap layers

`scripts/build_environment.py` enriches the prepared core. `scripts/build_data.py` calls it after the core build. Raw polygons, block points, OSM responses, queries and source hashes are cached under `data/raw/study/environment/`; missing or truncated downloads fail the build.

**Park access:** Retrieve TPL's published [public parks and 10-minute walk service areas](https://www.tpl.org/park-data-downloads) and its ParkServe place coverage. Retain service areas only for parks marked open access (`park_access=3`). Retrieve populated 2020 Census blocks and their internal points from the Census Bureau. For each tract, sum population whose block internal point intersects at least one service area, and divide by total 2020 block population. Overlaps count once. A primary `park_access_pct` is reported only if every populated block point is inside published ParkServe place coverage and the tract passes the existing 2020–2023 boundary review; otherwise it stays missing. `park_population_coverage_pct` records support separately. Tracts with zero 2020 population have no percentage.

This is our block-point approximation, **not TPL's official tract statistic or a current-population estimate**. It assumes people live at a block's internal point and cannot measure actual park use, safety, or quality. Published place coverage does not guarantee every park is recorded. The live service has legacy date labels: recorded retrieval dates do not assert that every polygon was updated that year. TPL's [method documentation](https://parkserve.tpl.org/downloads/ParkServe_Documentation_2026.pdf) and [source terms](https://parkserve.tpl.org/downloads/ParkServe_Terms_of_Use_May_2024.pdf) apply.

**Gathering places:** Retrieve OpenStreetMap nodes, ways and relations tagged `amenity=cafe`, `pub`, `library`, `community_centre`, `social_centre`, `place_of_worship`, or `marketplace` in each metro envelope. This is a documented study selection, not an exhaustive definition of public gathering places. Exclude explicit `access=no/private/members` and abandoned/disused features. Nodes use their coordinates; ways/relations use their bounding-box center. Assign points to the downloaded 2023 Census tract polygons; a boundary tie goes to the smallest tract GEOID. Deduplicate overlapping downloads by OSM element type and ID. Distinct OSM elements for the same physical venue can remain. `amenities_count` counts features; `amenities_per_1000` divides by ACS population (missing when population is unavailable or zero).

Zero means no matching feature was mapped, not that no venue exists. Some venues require payment or limit participation. Counts do not establish use, social contact, or walking access. OSM base timestamps are recorded per response and differ from the older health and ACS periods. Attribution: **© OpenStreetMap contributors**, [ODbL](https://www.openstreetmap.org/copyright). Tag meanings follow the [OSM amenity documentation](https://wiki.openstreetmap.org/wiki/Key:amenity).

Current enriched snapshot: **23,932 tracts with supported park-access estimates**, **92,159 mapped gathering-place features assigned to tracts**, and **29,722 tracts with a usable amenities-per-resident rate**. Park support and amenity rate availability are positive in all 20 metros; availability is not complete tract coverage. The OSM source base dates range from 2026-11-08 21:31:10 to 21:53:35 UTC. The original EPA/CDC/ACS columns and baseline analysis sample are unchanged.

## Census access

The build uses the Census Bureau’s official 2023 table-based ACS Summary Files. These downloads need no key and contain the same detailed-table estimates and margins of error as the keyed API. Direct API access has now been verified with the locally activated key. Eight raw/derived sample columns match the official bulk snapshot exactly; see `census-api-comparison.json`. The key is not included in any report or committed file.

To enable the direct API:

1. Register at [Census API key signup](https://api.census.gov/data/key_signup.html).
2. Follow the activation link Census sends to your email.
3. Add `CENSUS_API_KEY=your_key` to this repo’s local `.env`; keep it out of Git and chat.
4. Run `uv run python -m scripts.build_data --check-census` to validate it without printing the key.

## Fixed study definitions

- Rank metropolitan statistical areas by `POPESTIMATE2023` in the [Census Vintage 2023 population file](https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/metro/totals/cbsa-est2023-alldata.csv).
- Use the full constituent counties in that file, under July 2023 OMB delineations. Do not substitute central cities, metropolitan divisions, or micropolitan areas.
- Use 2023 ACS tract IDs and land areas, 2019–2023 ACS five-year estimates, and 2025-release CDC PLACES MHLTH crude prevalence based on BRFSS 2023.
- Use the EPA 2021 National Walkability Index, with its older source geography and 2018 source population. A shared ID label does not make the reference years equal.

## Population crosswalk review

The published source is [Amos (2021), 2020 Census Block Crosswalk Data, Harvard Dataverse V2](https://doi.org/10.7910/DVN/T9VMJO), available under CC BY 4.0. Its `bg19block20_pop_crosswalks` allocates 2019 block-group population shares to 2020 blocks using 2010 population and land-cover-informed interpolation. The companion block population file reallocates ACS 2015–2019 population. This review checks the documented method and its implementation; it is not independent peer review.

The build performs these checks:

1. Verify the unrestricted release files against the publisher’s checksums and record raw SHA256 hashes.
2. Match the EPA archive’s source block-group IDs to the published 2019 block-group IDs. EPA’s field name `GEOID20` is not treated as 2020 geography.
3. Sum full outgoing source shares, including out-of-scope targets, and reject deviations above 0.5%. Small rounding discrepancies are normalized across the full source.
4. Multiply shares by each EPA block group’s `TotPop` before averaging scores within target tracts. Outgoing shares alone are not target averaging weights.
5. Compare source support against the author’s modeled ACS2019 block population. A target with any unsupported positive-reference-population block loses its primary exposure; a partial diagnostic score remains available.
6. Require the 2020 and 2023 tract ID inventories to agree in the selected counties. Record land-area differences; inventory agreement does not prove polygon equality.
7. Exclude the primary exposure where absolute land-area change exceeds 1%, or the area comparison is undefined. The 1% threshold is a conservative study rule, not a validated accuracy threshold. Smaller or equal-area boundary shifts remain possible.

Observed checks: 77,295 EPA source block groups; **0 positive source population without weights**; maximum outgoing-share deviation 0.00007300; 89,404 source-to-target pairs. 5,117 tracts have some recorded land-area change; 245 fail the boundary gate, including undefined zero-land comparisons. These totals include tracts already missing other measures.

The complete review metrics and the 14 unpublished ACS tract IDs are in [the canonical audit](../../data/processed/study/audit.json). [Census erratum 148](https://www.census.gov/programs-surveys/acs/technical-documentation/errata/148.html) documents the Wyandanch/Brentwood collection-error suppression. No replacement population or income values are invented.

## Material health-data gap

The [CDC 2025 release notes](https://www.cdc.gov/places/current-release-notes/) say that 2023-BRFSS-based estimates are unavailable for Pennsylvania and Kentucky. Pennsylvania falls inside this study’s Philadelphia metro universe. Consequently, only **503 of Philadelphia’s 1,556 tracts** have the selected health outcome; those observations cover its non-Pennsylvania portions. They cannot represent the full Philadelphia metro. The dataset retains the missing Pennsylvania rows and adds state- and metro-level coverage diagnostics; no other release is substituted. A future sensitivity study could use an earlier release consistently across every metro.

## Coverage by metro

| Rank | Metro | Tracts | Mental distress | Primary walkability | Complete adjusted |
|---:|---|---:|---:|---:|---:|
| 1 | New York-Newark-Jersey City, NY-NJ | 4928 | 4817 | 4816 | 4719 |
| 2 | Los Angeles-Long Beach-Anaheim, CA | 3112 | 3086 | 3066 | 3039 |
| 3 | Chicago-Naperville-Elgin, IL-IN | 2292 | 2282 | 2267 | 2255 |
| 4 | Dallas-Fort Worth-Arlington, TX | 1704 | 1699 | 1681 | 1677 |
| 5 | Houston-Pasadena-The Woodlands, TX | 1615 | 1605 | 1604 | 1587 |
| 6 | Atlanta-Sandy Springs-Roswell, GA | 1504 | 1501 | 1462 | 1449 |
| 7 | Washington-Arlington-Alexandria, DC-VA-MD-WV | 1464 | 1455 | 1449 | 1437 |
| 8 | Philadelphia-Camden-Wilmington, PA-NJ-DE-MD | 1556 | 503 | 1526 | 499 |
| 9 | Miami-Fort Lauderdale-West Palm Beach, FL | 1497 | 1477 | 1454 | 1445 |
| 10 | Phoenix-Mesa-Chandler, AZ | 1104 | 1088 | 1088 | 1073 |
| 11 | Boston-Cambridge-Newton, MA-NH | 1134 | 1113 | 1110 | 1099 |
| 12 | Riverside-San Bernardino-Ontario, CA | 984 | 982 | 969 | 963 |
| 13 | San Francisco-Oakland-Fremont, CA | 1102 | 1092 | 1070 | 1059 |
| 14 | Detroit-Warren-Dearborn, MI | 1356 | 1297 | 1326 | 1277 |
| 15 | Seattle-Tacoma-Bellevue, WA | 863 | 860 | 849 | 848 |
| 16 | Minneapolis-St. Paul-Bloomington, MN-WI | 892 | 890 | 878 | 875 |
| 17 | Tampa-St. Petersburg-Clearwater, FL | 788 | 776 | 742 | 740 |
| 18 | San Diego-Chula Vista-Carlsbad, CA | 737 | 735 | 734 | 728 |
| 19 | Denver-Aurora-Centennial, CO | 707 | 698 | 685 | 680 |
| 20 | Baltimore-Columbia-Towson, MD | 717 | 709 | 709 | 701 |

## Reproduce and inspect

```sh
uv sync --extra build --extra dev
uv run python -m scripts.build_data
uv run python -m scripts.check
uv run pytest -q
```

The initial build downloads approximately 1 GB of cached public inputs to `data/raw/study` (ignored by Git and Docker). Keep the prepared `data/processed/study` snapshot with the app; its individual files are below GitHub’s 100 MB limit. `tracts.csv` is the agent table; `metros.csv`, `counties.csv`, `coverage.csv`, `crosswalk.csv`, `geography-comparison.csv`, `audit.json`, and `manifest.json` make the join reviewable. The manifest records hashes and column provenance.

Tract means and regression coefficients are not population-average individual effects. CDC outcomes are modeled and share demographic inputs with ACS controls. Crosswalk estimates, missingness, model uncertainty, source-year differences and small numbers of metro clusters limit interpretation. No adjacency file is included, so residual Moran diagnostics remain unavailable.

## Validation

All three agents have been exercised through the actual repo’s live `/chat` route on the real study. The saved tool traces are in `live-agent-checks.json`. Software checks include artifact hashes, support and boundary exclusions, source attribution, and structural-missingness disclosure.

Final validation: **40 software tests passed**, lint passed, and all three live agents returned HTTP 200 on the final real snapshot with the Pennsylvania caveat retained. Census API access is verified; eight sample columns match the bulk snapshot. See `live-agent-checks.json`, `census-api-comparison.json`, and `sample-comparisons.md`.

Runtime cleanup: the app and CLI load the real study by default and reject synthetic snapshots. Controlled numerical fixtures are created only in pytest temporary directories; they are excluded from the deployed image.

Code layout: `app.py` contains the agents and routes; `data.py` contains their deterministic data tools; `scripts/build_data.py` is the offline build. The runtime uses the same two-file pattern as the original starter.

The simplified repository uses two runtime files: `app.py` and `data.py`. The optional source reader uses GDAL attribute arrays without a spatial dataframe dependency. All three agents and the Census API were checked after restructuring.
