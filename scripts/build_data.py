"""Reproducible study: 20 largest 2023 metros, ACS, CDC, EPA, ParkServe and OSM.

Downloads are offline build inputs. No keys, accounts, or LLMs are needed for this build.
Raw sources are cached under data/raw/study; the small served snapshot is data/processed/study.
"""

import argparse
import csv
import os
import fcntl
import hashlib
import io
import json
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from data import ROOT, Catalog, digest, validate_tracts
from dotenv import load_dotenv

CDC_ID = "cwsq-ngmh"
CDC_RELEASE = "2025"
TARGET_GEOGRAPHY = "census2023-tract"
ACS_VARIABLES = {
    "B01003_001E": "population", "B19013_001E": "income", "B19013_001M": "income_moe",
    "B17001_001E": "poverty_denominator", "B17001_002E": "poverty_count",
    "B17001_002M": "poverty_count_moe", "B02001_003E": "black_count", "B03003_003E": "hispanic_count",
    **{f"B01001_{i:03d}E": f"age65_{i}" for i in [20, 21, 22, 23, 24, 25, 44, 45, 46, 47, 48, 49]},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_acs(rows: list[list[str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    frame["geoid"] = frame["state"] + frame["county"] + frame["tract"]
    for code, name in ACS_VARIABLES.items():
        # Census uses large negative sentinels for suppressed/unavailable estimates.
        values = pd.to_numeric(frame[code], errors="coerce")
        frame[name] = values.where(values >= 0)
    population = frame.population.where(frame.population > 0)
    frame["poverty_pct"] = 100 * frame.poverty_count / frame.poverty_denominator.where(frame.poverty_denominator > 0)
    frame["black_pct"] = 100 * frame.black_count / population
    frame["hispanic_pct"] = 100 * frame.hispanic_count / population
    frame["age65_pct"] = 100 * frame[[f"age65_{i}" for i in [20, 21, 22, 23, 24, 25, 44, 45, 46, 47, 48, 49]]].sum(axis=1, min_count=12) / population
    return frame[["geoid", "population", "income", "income_moe", "poverty_pct", "poverty_count_moe",
                  "black_pct", "hispanic_pct", "age65_pct"]]


POP_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/metro/totals/cbsa-est2023-alldata.csv"
ACS_BASE = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData"
TRACT23 = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2023/MapServer/8"
TRACT20 = "https://tigerweb.geo.census.gov/arcgis/rest/services/Census2020/tigerWMS_Census2020/MapServer/6"
UF_API = "https://dataverse.harvard.edu/api/access/datafile"
UF_DOI = "https://doi.org/10.7910/DVN/T9VMJO"
EPA_ARCHIVE = "https://edg.epa.gov/data/Public/OA/WalkabilityIndex.zip"


def request(url: str, params=None):
    for attempt in range(4):
        try:
            response = httpx.get(url, params=params, timeout=120, follow_redirects=True)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == 3:
                raise ValueError(f"Public source unavailable: {url.split('?')[0]}") from None
            time.sleep(2 ** attempt)


def download(raw: Path, name: str, url: str):
    path = raw / name
    if not path.exists():
        raw.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", url, timeout=180, follow_redirects=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(path)
    return path


def cache_json(raw: Path, name: str, url: str, params=None):
    path = raw / name
    if path.exists():
        return json.loads(path.read_text())
    value = request(url, params).json()
    if isinstance(value, dict) and "error" in value:
        raise ValueError(f"API error at {url}: {value['error']}")
    path.write_text(json.dumps(value) + "\n")
    return value


def select_metros(population: pd.DataFrame):
    metros = population[population.LSAD.eq("Metropolitan Statistical Area")].copy()
    if metros.CBSA.duplicated().any():
        raise ValueError("Metro ranking contains duplicate totals")
    metros = metros.sort_values(["POPESTIMATE2023", "CBSA"], ascending=[False, True]).head(20)
    if len(metros) != 20:
        raise ValueError("Ranking must contain twenty metros")
    metros = metros[["CBSA", "NAME", "POPESTIMATE2023"]].rename(
        columns={"CBSA": "metro", "NAME": "name", "POPESTIMATE2023": "population_2023"})
    metros.insert(0, "rank", range(1, 21))
    counties = population[population.CBSA.isin(metros.metro) & population.STCOU.notna()].copy()
    counties = counties[["CBSA", "STCOU", "NAME"]].rename(
        columns={"CBSA": "metro", "STCOU": "county", "NAME": "county_name"})
    # The official population CSV stores numeric FIPS without leading zeroes.
    counties["county"] = counties.county.str.zfill(5)
    if counties.county.duplicated().any() or not counties.county.str.fullmatch(r"\d{5}").all():
        raise ValueError("County membership is not unique or has malformed FIPS")
    return metros.reset_index(drop=True), counties.reset_index(drop=True)


def read_acs_bulk(raw: Path, counties: set[str]) -> pd.DataFrame:
    """Table-based Summary File cells are the same estimates/MOEs as the keyed API."""
    tables = sorted({code.split("_")[0].lower() for code in ACS_VARIABLES})
    merged = None
    for table in tables:
        path = raw / f"acs-{table}.dat"
        columns = {f"{code.split('_')[0]}_{code[-1]}{code.split('_')[1][:-1]}": code
                   for code in ACS_VARIABLES if code.lower().startswith(table + "_")}
        parts = []
        for chunk in pd.read_csv(path, sep="|", dtype=str, usecols=["GEO_ID", *columns], chunksize=50000):
            # Exact standard tract summary level, not tribal or geographic-variant rows.
            tract = chunk.GEO_ID.str.fullmatch(r"1400000US\d{11}")
            selected = chunk[tract & chunk.GEO_ID.str[9:14].isin(counties)].copy()
            selected["geoid"] = selected.GEO_ID.str[9:]
            parts.append(selected.drop(columns="GEO_ID").rename(columns=columns))
        table_frame = pd.concat(parts, ignore_index=True)
        validate_tracts(table_frame)
        merged = table_frame if merged is None else merged.merge(table_frame, on="geoid", how="outer", validate="one_to_one")
        print(f"ACS {table}: {len(table_frame)} selected tracts", flush=True)
    merged["state"] = merged.geoid.str[:2]
    merged["county"] = merged.geoid.str[2:5]
    merged["tract"] = merged.geoid.str[5:]
    headers = list(ACS_VARIABLES) + ["state", "county", "tract"]
    return parse_acs([headers, *merged[headers].values.tolist()])


def arcgis_inventory(raw: Path, name: str, url: str, where: str, fields: str, page_size=1000):
    combined = raw / (name + ".json")
    if combined.exists():
        saved = json.loads(combined.read_text())
        return pd.DataFrame(saved["records"])
    count = cache_json(raw, name + "-count.json", url + "/query",
                       {"where": where, "returnCountOnly": "true", "f": "json"})["count"]
    records = []
    while len(records) < count:
        offset = len(records)
        page = cache_json(raw, f"{name}-{offset}.json", url + "/query", {
            "where": where, "outFields": fields, "orderByFields": "OBJECTID", "returnGeometry": "false",
            "resultOffset": offset, "resultRecordCount": page_size, "f": "json",
        })
        features = page.get("features", [])
        if not features:
            raise ValueError(f"Truncated ArcGIS pagination for {name}: {len(records)} / {count}")
        records.extend(f["attributes"] for f in features)
    if len(records) != count:
        raise ValueError(f"ArcGIS count changed during download for {name}")
    combined.write_text(json.dumps({"url": url, "where": where, "fields": fields,
                                    "expected_count": count, "records": records}) + "\n")
    return pd.DataFrame(records)


def get_geography_and_epa(raw: Path, counties: pd.DataFrame):
    # Read only four attributes directly from the official compressed geodatabase.
    # Geometry is never extracted; the slow live EPA query service is unnecessary.
    attribute_path = raw / "epa-national-attributes.csv"
    if not attribute_path.exists():
        from pyogrio.raw import read
        path = "/vsizip/" + str((raw / "epa-nwi.zip").resolve()) + "/Natl_WI.gdb"
        metadata, _, _, values = read(path, columns=["GEOID10", "GEOID20", "TotPop", "NatWalkInd"], read_geometry=False)
        attributes = pd.DataFrame(dict(zip(metadata["fields"], values)))
        attributes.to_csv(attribute_path, index=False)
    national = pd.read_csv(attribute_path, dtype={"GEOID10": str, "GEOID20": str})
    epa = national[national.GEOID20.str[:5].isin(set(counties.county))].copy()

    def geography_job(vintage, url):
        d = arcgis_inventory(raw, f"tract{vintage}-national", url, "1=1", "GEOID,AREALAND", page_size=100000)
        d = d.rename(columns={"GEOID": "geoid", "AREALAND": f"land_area_{vintage}"})
        d = d[d.geoid.str[:5].isin(set(counties.county))].copy()
        validate_tracts(d)
        print(f"TIGER {vintage}: {len(d)} selected tracts", flush=True)
        return d

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(geography_job, 2023, TRACT23)
        b = pool.submit(geography_job, 2020, TRACT20)
        tract23, tract20 = a.result(), b.result()
    print(f"EPA archive: {len(epa)} selected block groups", flush=True)
    if epa.GEOID20.duplicated().any():
        raise ValueError("EPA's service source-geography IDs are not unique")
    if not epa.GEOID20.str.fullmatch(r"\d{12}").all():
        raise ValueError("EPA source block group IDs must have 12 digits")
    for col in ["TotPop", "NatWalkInd"]:
        epa[col] = pd.to_numeric(epa[col], errors="raise")
    if epa.TotPop.isna().any() or (epa.TotPop < 0).any():
        raise ValueError("Invalid EPA source population")
    if not epa.NatWalkInd.dropna().between(1, 20).all():
        raise ValueError("Invalid EPA index")
    validate_tracts(tract23)
    validate_tracts(tract20)
    return epa, tract23, tract20


def get_places(raw: Path, counties: set[str]):
    meta = cache_json(raw, "cdc-metadata.json", f"https://chronicdata.cdc.gov/api/views/{CDC_ID}.json")
    if f"{CDC_RELEASE} release" not in meta.get("name", "") or "Census Tract" not in meta.get("name", ""):
        raise ValueError("CDC release identity changed")
    where = "countyfips in (" + ",".join(f"'{c}'" for c in sorted(counties)) + ") AND measureid='MHLTH' AND datavaluetypeid='CrdPrv'"
    url = f"https://chronicdata.cdc.gov/resource/{CDC_ID}.json"
    expected = int(cache_json(raw, "cdc-count.json", url, {"$where": where, "$select": "count(*)"})[0]["count"])
    records = []
    while len(records) < expected:
        page = cache_json(raw, f"cdc-{len(records)}.json", url, {
            "$where": where, "$select": "locationid,year,data_value,low_confidence_limit,high_confidence_limit",
            "$order": "locationid", "$limit": 10000, "$offset": len(records),
        })
        if not page:
            raise ValueError("CDC download ended before the expected count")
        records.extend(page)
    if len(records) != expected or {str(r["year"]) for r in records} != {"2023"}:
        raise ValueError("CDC count or observation year changed")
    frame = pd.DataFrame([{"geoid": r["locationid"], "distress_pct": r.get("data_value"),
                           "distress_ci_low": r.get("low_confidence_limit"),
                           "distress_ci_high": r.get("high_confidence_limit")} for r in records])
    validate_tracts(frame)
    print(f"CDC PLACES: {len(frame)} tracts", flush=True)
    return frame


def parse_crosswalk_row(row: list[str]):
    if len(row) % 2 != 1 or len(row[0]) != 15:
        raise ValueError("Malformed university block crosswalk row")
    pairs = []
    for i in range(1, len(row), 2):
        bg, share = row[i], float(row[i + 1])
        if len(bg) != 12 or not bg.isdigit() or not np.isfinite(share) or not 0 <= share <= 1:
            raise ValueError("Invalid crosswalk source ID or population share")
        pairs.append((bg, share))
    return row[0], pairs


def review_boundaries(tract23: pd.DataFrame, tract20: pd.DataFrame):
    if set(tract23.geoid) != set(tract20.geoid):
        raise ValueError("2020–2023 tract IDs changed; a further crosswalk is required")
    comparison = tract23.merge(tract20, on="geoid", validate="one_to_one")
    comparison["land_area_change_m2"] = comparison.land_area_2023.astype(float) - comparison.land_area_2020.astype(float)
    comparison["boundary_land_area_change_pct"] = 100 * comparison.land_area_change_m2.abs() / comparison.land_area_2020.where(comparison.land_area_2020 > 0)
    # Conservative study-specific QC, not evidence that polygons below the threshold are identical.
    comparison["boundary_review_excluded"] = (~comparison.boundary_land_area_change_pct.le(1)).astype(int)
    return comparison


def crosswalk_weights(raw: Path, epa: pd.DataFrame, counties: set[str]):
    """Multiply FULL-source normalized population shares by EPA source population.

    A source share is not itself a target averaging weight. Keep all outgoing shares
    for normalization, including blocks outside the target universe.
    """
    populations = epa.set_index("GEOID20").TotPop.to_dict()
    scores = epa.set_index("GEOID20").NatWalkInd.to_dict()
    states = sorted({c[:2] for c in counties})
    pair_shares, source_totals = defaultdict(float), defaultdict(float)
    covered_blocks, target_blocks = set(), set()
    crosswalk = zipfile.ZipFile(raw / "uf-bg19-block20.zip")
    for state in states:
        with crosswalk.open(f"bg19block20_pop_crosswalk_{state}.csv") as stream:
            for row in csv.reader(io.TextIOWrapper(stream)):
                block, pairs = parse_crosswalk_row(row)
                target = block[:11]
                in_scope = block[:5] in counties
                if in_scope:
                    target_blocks.add(block)
                    # A missing positive-share source invalidates support for that block.
                    positive = [(bg, share) for bg, share in pairs if share > 0]
                    if positive and all(bg in populations and populations[bg] > 0 and pd.notna(scores[bg]) for bg, _ in positive):
                        covered_blocks.add(block)
                for bg, share in pairs:
                    if bg in populations:
                        source_totals[bg] += share
                        if in_scope and share > 0:
                            pair_shares[(bg, target)] += share
    weights = pd.DataFrame([(s, t, share) for (s, t), share in pair_shares.items()],
                           columns=["source_bg", "target_geoid", "raw_share"])
    weights["source_share_sum"] = weights.source_bg.map(source_totals)
    # Author's rounded shares can differ slightly from one; never normalize a substantive loss.
    bad = {bg: total for bg, total in source_totals.items() if total > 0 and abs(total - 1) > .005}
    if bad:
        raise ValueError(f"Crosswalk outgoing shares do not sum to one: {list(bad.items())[:5]}")
    weights["population_weight"] = weights.raw_share / weights.source_share_sum * weights.source_bg.map(populations)
    weights["walkability"] = weights.source_bg.map(scores)
    weights["weighted_score"] = weights.population_weight * weights.walkability
    good = weights.walkability.notna() & weights.population_weight.gt(0)
    totals = weights[good].groupby("target_geoid")[["weighted_score", "population_weight"]].sum()
    result = (totals.weighted_score / totals.population_weight).rename("walkability_partial").rename_axis("geoid").reset_index()
    # Independent block-population support diagnostic, using author's ACS2019 block estimates.
    block_population = zipfile.ZipFile(raw / "uf-acs19-block20-pop.zip")
    support = defaultdict(lambda: [0, 0])
    for state in states:
        with block_population.open(f"pop19_{state}.csv") as stream:
            for block, population in csv.reader(io.TextIOWrapper(stream)):
                if block[:5] in counties:
                    value = int(population)
                    if value < 0:
                        raise ValueError("Negative block population")
                    support[block[:11]][0] += value
                    if block in covered_blocks:
                        support[block[:11]][1] += value
    support_frame = pd.DataFrame([(g, p, c) for g, (p, c) in support.items()],
        columns=["geoid", "crosswalk_reference_population", "supported_population"])
    support_frame["epa_population_coverage_pct"] = 100 * support_frame.supported_population / support_frame.crosswalk_reference_population.where(support_frame.crosswalk_reference_population > 0)
    result = support_frame.merge(result, on="geoid", how="outer", validate="one_to_one")
    # Main analysis excludes targets with any unsupported positive-population blocks.
    result["walkability"] = result.walkability_partial.where(result.epa_population_coverage_pct >= 100 - 1e-9)
    unmatched = [bg for bg, pop in populations.items() if pop > 0 and source_totals.get(bg, 0) == 0]
    audit = {"source_block_groups": len(epa), "positive_population_source_groups": sum(p > 0 for p in populations.values()),
             "source_groups_without_positive_weights": unmatched,
             "source_population_without_weights": float(sum(populations[bg] for bg in unmatched)),
             "max_outgoing_share_deviation": max((abs(v - 1) for v in source_totals.values() if v > 0), default=0),
             "strict_complete_support_tracts": int(result.walkability.notna().sum()),
             "partial_score_tracts": int(result.walkability_partial.notna().sum()),
             "crosswalk_pairs": len(weights), "positive_mass": float(weights.population_weight.sum())}
    return result.drop(columns="supported_population"), weights, audit


def source_record(identifier, title, url, release, geography, files, notes):
    hashes = {p.name: digest(p) for p in files}
    aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"id": identifier, "title": title, "url": url, "release": release, "geography": geography,
            "retrieved_at": now(), "sha256": aggregate, "notes": notes + ["Raw file SHA256s: " + json.dumps(hashes, sort_keys=True)]}


def prepare_inputs(raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    items = {"metro-population.csv": POP_URL, "uf-bg19-block20.zip": UF_API + "/4513654",
             "uf-acs19-block20-pop.zip": UF_API + "/4513653", "uf-readme.txt": UF_API + "/4575858",
             "epa-nwi.zip": EPA_ARCHIVE}
    for table in sorted({code.split("_")[0].lower() for code in ACS_VARIABLES}):
        items[f"acs-{table}.dat"] = f"{ACS_BASE}/acsdt5y2023-{table}.dat"
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, raw, name, url) for name, url in items.items()]
        for future in as_completed(futures):
            future.result()
    # Verify unrestricted university artifacts against publisher-provided checksums.
    metadata = cache_json(raw, "uf-crosswalk-metadata.json",
        "https://dataverse.harvard.edu/api/datasets/:persistentId/", {"persistentId": "doi:10.7910/DVN/T9VMJO"})
    files = {f["dataFile"]["id"]: f for f in metadata["data"]["latestVersion"]["files"]}
    for identifier, name in [(4513654, "uf-bg19-block20.zip"), (4513653, "uf-acs19-block20-pop.zip")]:
        file = files[identifier]
        if file.get("restricted"):
            raise ValueError("Crosswalk release is no longer unrestricted")
        expected = file["dataFile"]["checksum"]
        actual = digest(raw / name, expected["type"].lower())
        if actual != expected["value"]:
            raise ValueError(f"Publisher checksum mismatch: {name}")
    artifacts = {}
    for name, url in items.items():
        path = raw / name
        artifacts[name] = {"url": url, "sha256": digest(path), "bytes": path.stat().st_size,
                           "retrieved_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    registry = raw / "artifacts.json"
    if registry.exists():
        prior = json.loads(registry.read_text())
        for name in prior:
            if name in artifacts and artifacts[name]["sha256"] != prior[name]["sha256"]:
                raise ValueError(f"Cached source changed: {name}")
    registry.write_text(json.dumps(artifacts, indent=2) + "\n")


def build_study(root: Path):
    raw, output = root / "data/raw/study", root / "data/processed/study"
    print("Checking cached source artifacts and publisher checksums", flush=True)
    prepare_inputs(raw)
    population = pd.read_csv(raw / "metro-population.csv", encoding="latin1", dtype={"CBSA": str, "MDIV": str, "STCOU": str})
    metros, counties = select_metros(population)
    county_set = set(counties.county)
    if any(c.startswith("09") for c in county_set):
        raise ValueError("Connecticut's 2022 ID changes require a separate reviewed crosswalk")
    acs = read_acs_bulk(raw, county_set)
    with ThreadPoolExecutor(max_workers=2) as pool:
        geography_job = pool.submit(get_geography_and_epa, raw, counties)
        health_job = pool.submit(get_places, raw, county_set)
        epa, tract23, tract20 = geography_job.result()
        health = health_job.result()
    # Explicit inventory check before using a 2020-block parent ID for 2023 tracts.
    if set(tract23.geoid) != set(tract20.geoid):
        added, removed = set(tract23.geoid) - set(tract20.geoid), set(tract20.geoid) - set(tract23.geoid)
        raise ValueError(f"2020–2023 tract inventory changed; review required: added {sorted(added)[:10]}, removed {sorted(removed)[:10]}")
    if not set(acs.geoid) <= set(tract23.geoid):
        raise ValueError("ACS includes tract IDs outside the 2023 TIGER inventory")
    unpublished = sorted(set(tract23.geoid) - set(acs.geoid))
    geography_compare = review_boundaries(tract23, tract20)
    print("Building population-mass crosswalk and support diagnostics", flush=True)
    exposure, weights, audit = crosswalk_weights(raw, epa, county_set)
    target_missing = sorted(set(exposure.geoid) - set(tract23.geoid))
    if target_missing:
        raise ValueError(f"Crosswalk targets not in 2023 inventory: {target_missing[:10]}")
    membership = tract23[["geoid"]].copy()
    membership["county"] = membership.geoid.str[:5]
    membership = membership.merge(counties[["county", "metro"]], on="county", validate="many_to_one")
    frame = membership[["geoid", "metro"]].merge(acs, on="geoid", how="left", validate="one_to_one")
    frame = frame.merge(health, on="geoid", how="left", validate="one_to_one")
    frame = frame.merge(exposure, on="geoid", how="left", validate="one_to_one")
    frame = frame.merge(tract23, on="geoid", how="left", validate="one_to_one")
    frame = frame.merge(geography_compare[["geoid", "boundary_land_area_change_pct", "boundary_review_excluded"]],
                        on="geoid", how="left", validate="one_to_one")
    frame["walkability"] = frame.walkability.where(frame.boundary_review_excluded.eq(0))
    frame["pop_density"] = frame.population / (frame.land_area_2023.where(frame.land_area_2023 > 0) / 1e6)
    for col in frame.columns.difference(["geoid", "metro"]):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    validate_tracts(frame)
    sources = [
        source_record("census-metros-2023", "Census Vintage 2023 metro population and county membership", POP_URL,
            "July 1, 2023 population estimates; July 2023 OMB delineations", "2023-CBSA-and-counties", [raw / "metro-population.csv"],
            ["Top twenty metropolitan statistical areas ranked by POPESTIMATE2023; metropolitan divisions and micropolitan areas excluded."]),
        source_record("acs5-2023-bulk", "ACS 2019–2023 five-year Detailed Tables, official bulk files", ACS_BASE,
            "2023 ACS five-year", TARGET_GEOGRAPHY, sorted(raw.glob("acs-*.dat")),
            ["Official bulk cells match the keyed Census API; no API key is used for these downloads.",
             "Period estimates for 2019–2023; selected raw 90% margins of error retained, not propagated.",
             "Fourteen Suffolk County NY tract rows are unpublished; Census collection-error suppression is documented at https://www.census.gov/programs-surveys/acs/technical-documentation/errata/148.html . These rows remain missing.",
             "Poverty uses the population for whom poverty status is determined; Black alone and Hispanic of any race overlap."]),
        source_record("cdc-places-2025", "CDC PLACES frequent mental distress, crude prevalence", f"https://chronicdata.cdc.gov/resource/{CDC_ID}.json",
            "2025 release; BRFSS MHLTH 2023", TARGET_GEOGRAPHY, sorted(raw.glob("cdc-[0-9]*.json")),
            ["Modeled estimates, not direct tract observations; BRFSS/ACS/census inputs.",
             "The 2025 release omits Pennsylvania and Kentucky estimates for 2023-BRFSS-based measures. Pennsylvania's omission leaves the Philadelphia metro partially observed; no older-release values are substituted. https://www.cdc.gov/places/current-release-notes/",
             "Frequent mental distress: at least 14 mentally unhealthy days in the previous 30 days, adults 18+.",
             "Does not directly measure loneliness or social connection, and cannot evaluate intervention effects."]),
        source_record("epa-nwi-2021", "EPA National Walkability Index", EPA_ARCHIVE,
            "2021 index; SLD 2019 geography; service GEOID20 labeled 2018; population TotPop 2018", "EPA-source-block-groups", [raw / "epa-nwi.zip", raw / "epa-national-attributes.csv"],
            ["Index is relative walkability on a 1–20 scale, not measured walking, safety, or social interaction.",
             "GEOID20 is a service field name, not a claim of 2020 geography; source IDs are matched to the author's 2019-block-group crosswalk."]),
        source_record("amos-crosswalk-v2", "Amos (2021), 2020 Census Block Crosswalk Data, V2", UF_DOI,
            "V2, CC BY 4.0; 2019 block groups → 2020 blocks", "2019-BG-to-2020-blocks", [raw / "uf-bg19-block20.zip", raw / "uf-acs19-block20-pop.zip", raw / "uf-readme.txt"],
            ["Crosswalk uses 2010 population and NLCD2016-informed interpolation within split blocks; it is an estimate, not observed 2023 population movement.",
             "Outgoing population shares are normalized only within 0.5% rounding tolerance, then multiplied by each EPA block group's 2018 TotPop to form population masses.",
             "2020 block parent tract IDs are validated against both 2020 and 2023 tract inventories; this is not proof of exact polygon equality.",
             "Main walkability additionally excludes targets with more than 1% land-area change between 2020 and 2023 or an undefined area comparison. This is a study QC rule, not a validated accuracy threshold.",
             "Main walkability excludes tracts with unsupported positive-population blocks in the author's reallocated ACS2019 reference population. Partial scores are retained separately.",
             "Attribution: Brian Amos (2021), doi:10.7910/DVN/T9VMJO, Harvard Dataverse V2; CC BY 4.0."]),
        source_record("census-tiger-2023", "Census TIGERweb 2023 ACS tract land areas", TRACT23,
            "2023 ACS geography, compared with Census 2020 tract inventory", TARGET_GEOGRAPHY, [raw / "tract2023-national.json", raw / "tract2020-national.json"],
            ["Population density uses ACS population / Census AREALAND in square kilometers.",
             "Inventory agreement does not guarantee identical 2020/2023 polygon boundaries; land-area differences are recorded in geography-comparison.csv."]),
    ]
    column_sources = {c: ["acs5-2023-bulk"] for c in acs if c != "geoid"}
    for c in health:
        if c != "geoid":
            column_sources[c] = ["cdc-places-2025"]
    for c in exposure:
        if c != "geoid":
            column_sources[c] = ["epa-nwi-2021", "amos-crosswalk-v2", "census-tiger-2023"]
    column_sources["land_area_2023"] = ["census-tiger-2023"]
    column_sources["boundary_land_area_change_pct"] = ["census-tiger-2023"]
    column_sources["boundary_review_excluded"] = ["census-tiger-2023"]
    column_sources["pop_density"] = ["acs5-2023-bulk", "census-tiger-2023"]
    output.mkdir(parents=True, exist_ok=True)
    frame.sort_values("geoid").to_csv(output / "tracts.csv", index=False)
    metros.to_csv(output / "metros.csv", index=False)
    counties.to_csv(output / "counties.csv", index=False)
    weights.to_csv(output / "crosswalk.csv", index=False)
    geography_compare.to_csv(output / "geography-comparison.csv", index=False)
    required = ["walkability", "distress_pct", "poverty_pct", "income", "age65_pct", "black_pct", "hispanic_pct", "pop_density"]
    coverage = frame.groupby("metro").agg(tracts=("geoid", "size"), walkability=("walkability", "count"), distress=("distress_pct", "count"), acs_population=("population", "sum"))
    complete = frame[required].notna().all(axis=1) & frame.income.gt(0)
    coverage["complete_adjusted"] = frame.assign(complete=complete).groupby("metro").complete.sum()
    coverage = metros.merge(coverage, on="metro", validate="one_to_one")
    coverage["health_coverage_pct"] = 100 * coverage.distress / coverage.tracts
    coverage["complete_adjusted_coverage_pct"] = 100 * coverage.complete_adjusted / coverage.tracts
    coverage.to_csv(output / "coverage.csv", index=False)
    state_coverage = frame.assign(state_fips=frame.geoid.str[:2]).groupby("state_fips").agg(
        tracts=("geoid", "size"), health_available=("distress_pct", "count")).reset_index()
    state_coverage.to_csv(output / "state-coverage.csv", index=False)
    structural_missing = state_coverage[state_coverage.health_available.eq(0)].to_dict(orient="records")
    audit.update({"2020_2023_same_tract_id_inventory": True,
                  "tracts_with_land_area_changes": int(geography_compare.land_area_change_m2.ne(0).sum()),
                  "max_abs_land_area_change_m2": float(geography_compare.land_area_change_m2.abs().max()),
                  "tracts_excluded_by_boundary_review": int(geography_compare.boundary_review_excluded.sum()),
                  "boundary_rule": "Exclude main exposure if absolute land-area change exceeds 1% or comparison is undefined.",
                  "final_nonmissing_walkability_tracts": int(frame.walkability.notna().sum()),
                  "epa_population_mass_outside_selected_counties": float(epa.TotPop.sum() - weights.population_weight.sum()),
                  "health_rows_outside_universe": int((~health.geoid.isin(frame.geoid)).sum()),
                  "tracts_without_published_acs_rows": unpublished,
                  "tracts": len(frame), "metros": int(frame.metro.nunique()), "counties": len(counties),
                  "complete_adjusted_tracts": int(complete.sum()),
                  "complete_adjusted_metros": int(frame.loc[complete, "metro"].nunique()),
                  "states_with_no_health_estimates": structural_missing,
                  "missing_by_column": {c: int(frame[c].isna().sum()) for c in frame.columns},
                  "review_status": "Documented methodology and automated checks; not independent peer review."})
    manifest = {"snapshot": "top20-2023-core-" + now(), "synthetic": False, "geography_id": TARGET_GEOGRAPHY,
        "files": {p.name: digest(p) for p in output.glob("*.csv")}, "sources": sources, "column_sources": column_sources,
        "scope": {"description": "Twenty largest US metropolitan statistical areas by Census Vintage 2023 population; full constituent counties, July 2023 OMB delineations.",
                  "ranking_year": 2023, "population_field": "POPESTIMATE2023", "top20_verified": True,
                  "structural_health_gaps": structural_missing,
                  "metro_count": 20, "county_count": len(counties), "metro_lookup": metros.to_dict(orient="records"),
                  "coverage": coverage.to_dict(orient="records")},
        "definitions": {"distress_pct": "Modeled crude adult frequent mental-distress prevalence (%).",
                        "walkability": "Population-mass weighted EPA index on a 1–20 scale, strict support sample.",
                        "walkability_partial": "Diagnostic EPA score before excluding incomplete crosswalk support; not the primary exposure.",
                        "epa_population_coverage_pct": "Percent of author's modeled ACS2019 block population supported by usable EPA sources.",
                        "pop_density": "ACS residents per square kilometer of Census land area.",
                        "park_access_pct": "Not loaded in this core study snapshot.",
                        "amenities_per_1000": "Not loaded in this core study snapshot."},
        "caveats": ["Cross-sectional tract associations cannot establish causation or individual effects.",
                    "The study combines older built-environment measures, interpolated population weights, ACS period estimates, and modeled 2023 health outcomes.",
                    "Crosswalk review is documented and checked in code, not an independent validation of interpolated weights.",
                    "Minor boundary/cartographic changes can remain even when 2020 and 2023 tract ID inventories agree.",
                    "CDC PLACES 2025 omits Pennsylvania's 2023 frequent-mental-distress estimates. Philadelphia health/model samples cover only available non-Pennsylvania portions, not the full metro. Do not generalize those subset results to the whole metro. See https://www.cdc.gov/places/current-release-notes/ .",
                    "No direct loneliness or social-connection measures; park access and OSM amenities are not yet loaded.",
                    "No tract adjacency file is included; residual Moran diagnostics are unavailable, while metro clustering remains available.",
                    "Missing observations remain missing; the analyst must disclose complete-case exclusions."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    catalog = Catalog(output)
    if catalog.frame.metro.nunique() != 20 or frame.loc[complete, "metro"].nunique() != 20:
        raise ValueError("Not all twenty metros have complete adjusted observations")
    print(json.dumps({k: audit[k] for k in ["tracts", "metros", "counties", "complete_adjusted_tracts"]}), flush=True)
    from scripts.build_environment import enrich_study
    return enrich_study(root)


def check_census():
    """Check local API access against the prepared table without logging the key."""
    key = os.getenv("CENSUS_API_KEY")
    if not key:
        raise SystemExit("Set CENSUS_API_KEY in .env after activating it at https://api.census.gov/data/key_signup.html")
    try:
        response = httpx.get("https://api.census.gov/data/2023/acs/acs5", timeout=60, follow_redirects=True,
            params={"get": ",".join(ACS_VARIABLES), "for": "tract:002400", "in": "state:36 county:061", "key": key})
        response.raise_for_status()
        sample = parse_acs(response.json()).iloc[0]
        actual = Catalog(ROOT / "data/processed/study").frame.set_index("geoid").loc[sample.geoid]
        columns = [c for c in sample.index if c != "geoid"]
        if not all(np.isclose(float(sample[c]), float(actual[c]), equal_nan=True) for c in columns):
            raise ValueError("Sample mismatch")
    except Exception:
        raise SystemExit("Census check failed; check activation, connectivity, and source availability. No key value was logged.") from None
    print(f"Census API verified; {len(columns)} columns match the study snapshot.")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Prepare the real study from public sources")
    parser.add_argument("--check-census", action="store_true", help="Verify your local API key against the prepared data")
    args = parser.parse_args()
    if args.check_census:
        check_census()
    else:
        raw = ROOT / "data/raw/study"
        raw.mkdir(parents=True, exist_ok=True)
        with (raw / ".build.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("A data build is already running") from None
            print(build_study(ROOT))
