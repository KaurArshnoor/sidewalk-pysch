"""Add real ParkServe access and OSM gathering places to the prepared study.

Build-only spatial dependencies stay out of the running agents. Cached public downloads
make interrupted builds resumable; no missing download is ever converted to a zero.
"""
import argparse
import fcntl
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from data import ROOT, Catalog, digest
from scripts.build_data import TRACT23, arcgis_inventory, cache_json, now, source_record

BLOCK20 = "https://tigerweb.geo.census.gov/arcgis/rest/services/Census2020/tigerWMS_Census2020/MapServer/10"
PARK_BASE = "https://server7.tplgis.org/arcgis7/rest/services/ParkServe"
PARK_AREAS = PARK_BASE + "/ParkServe_Shareable/MapServer/1"
PARKS = PARK_BASE + "/ParkServe_Shareable/MapServer/0"
PARK_PLACES = PARK_BASE + "/ParkServe_ProdNew/MapServer/1"
OVERPASS = "https://overpass-api.de/api/interpreter"
AMENITIES = ("cafe", "pub", "library", "community_centre", "social_centre", "place_of_worship", "marketplace")
PARK_DEFINITION = (
    "Estimated percent of 2020 Census population whose block internal point falls in a published "
    "ParkServe 10-minute walk service area for an open-access park. Only tracts with all populated "
    "block points inside the published ParkServe place coverage and passing the existing boundary "
    "review are reported. This is a study-derived block-point approximation, not TPL's official statistic."
)
OSM_DEFINITION = (
    "Mapped OSM gathering-place features per 1,000 ACS residents: " + ", ".join(AMENITIES) + ". "
    "Nodes use their coordinates; ways/relations use their bounding-box centers. Explicit private, "
    "no-access and members-only features are excluded. Counts are mapped features, not verified unique venues."
)


def features(raw: Path, name: str, url: str, *, where="1=1", bbox=None, fields="*"):
    """Freeze object IDs before paging; fail if any requested feature is missing."""
    complete = raw / (name + ".geojson")
    if complete.exists():
        return json.loads(complete.read_text())["features"]
    spatial = {} if bbox is None else {
        "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
    }
    ids = cache_json(raw, name + "-ids.json", url + "/query",
                     {"f": "json", "where": where, "returnIdsOnly": "true", **spatial})
    identifiers = sorted(ids.get("objectIds") or [])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate source IDs: {name}")
    records = []
    for offset in range(0, len(identifiers), 200):
        batch = identifiers[offset:offset + 200]
        page = cache_json(raw, f"{name}-{offset}.geojson", url + "/query", {
            "f": "geojson", "objectIds": ",".join(map(str, batch)), "outFields": fields,
            "returnGeometry": "true", "outSR": "4326",
        })
        fetched = page.get("features", [])
        oid = ids["objectIdFieldName"]
        if {f["properties"][oid] for f in fetched} != set(batch):
            raise ValueError(f"Incomplete geometry page: {name}/{offset}")
        if any(not f.get("geometry") for f in fetched):
            raise ValueError(f"Missing source geometry: {name}")
        records.extend(fetched)
    complete.write_text(json.dumps({"type": "FeatureCollection", "url": url,
                                   "retrieved_at": now(), "features": records}) + "\n")
    return records


def census_county(raw, county):
    where = f"STATE='{county[:2]}' AND COUNTY='{county[2:]}'"
    tracts = features(raw, "tracts-" + county, TRACT23, where=where, fields="GEOID,OBJECTID")
    blocks = arcgis_inventory(raw, "blocks-" + county, BLOCK20, where + " AND POP100>0",
                             "GEOID,POP100,INTPTLAT,INTPTLON", page_size=2000)
    return tracts, blocks


def membership(points, polygons):
    """True for points covered by any polygon, including its boundary; overlap counts once."""
    from shapely import STRtree
    if not len(polygons):
        return np.zeros(len(points), dtype=bool)
    pairs = STRtree(polygons).query(points, predicate="intersects")
    covered = np.zeros(len(points), dtype=bool)
    covered[np.unique(pairs[0])] = True
    return covered


def park_access(blocks, places, service_areas, tracts):
    from shapely import points
    if blocks.GEOID.duplicated().any() or not blocks.GEOID.str.fullmatch(r"\d{15}").all():
        raise ValueError("Malformed or duplicate Census block IDs")
    pop = pd.to_numeric(blocks.POP100, errors="raise")
    xy = blocks[["INTPTLON", "INTPTLAT"]].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(xy).all() or not (pop > 0).all():
        raise ValueError("Invalid populated-block coordinates or population")
    if not set(blocks.GEOID.str[:11]) <= set(tracts.geoid):
        raise ValueError("2020 block parents do not match the reviewed 2023 tract inventory")
    pts = points(xy)
    inside = membership(pts, places)
    served = membership(pts, service_areas)
    b = pd.DataFrame({"geoid": blocks.GEOID.str[:11], "park_population_2020": pop,
                      "park_covered_population_2020": pop * inside,
                      "park_served_population_2020": pop * (served & inside)})
    result = tracts[["geoid", "boundary_review_excluded"]].merge(
        b.groupby("geoid").sum(), on="geoid", how="left", validate="one_to_one")
    pop_columns = ["park_population_2020", "park_covered_population_2020", "park_served_population_2020"]
    result[pop_columns] = result[pop_columns].fillna(0)
    denominator = result.park_population_2020.where(result.park_population_2020 > 0)
    result["park_population_coverage_pct"] = 100 * result.park_covered_population_2020 / denominator
    supported = result.park_covered_population_2020.eq(result.park_population_2020)
    supported &= result.boundary_review_excluded.eq(0)
    result["park_access_pct"] = (100 * result.park_served_population_2020 / denominator).where(supported)
    return result.drop(columns="boundary_review_excluded")


def osm_query(raw, metro, bbox):
    path = raw / f"osm-{metro}.json"
    if path.exists():
        value = json.loads(path.read_text())
    else:
        west, south, east, north = bbox
        query = ('[out:json][timeout:180];nwr[amenity~"^(' + "|".join(AMENITIES)
                 + f')$"]({south},{west},{north},{east});out center;')
        for attempt in range(4):
            try:
                response = httpx.get(OVERPASS, params={"data": query}, timeout=210, follow_redirects=True,
                    headers={"User-Agent": "SidewalkPsychResearch/0.1 (https://github.com/KaurArshnoor/sidewalk-pysch)",
                             "Accept": "application/json"})
                response.raise_for_status()
                value = response.json()
                if value.get("remark") or "elements" not in value or not value.get("osm3s", {}).get("timestamp_osm_base"):
                    raise ValueError("Overpass returned an incomplete query")
                break
            except (httpx.HTTPError, ValueError) as exc:
                print(f"OSM metro {metro}: attempt {attempt + 1} failed ({type(exc).__name__})", flush=True)
                if attempt == 3:
                    raise ValueError(f"OSM query failed for metro {metro}; no zeros were written") from None
                time.sleep(30 * (attempt + 1))
        value["query"] = query
        value["retrieved_at"] = now()
        path.write_text(json.dumps(value) + "\n")
    if value.get("remark") or "elements" not in value:
        raise ValueError(f"Incomplete cached OSM response: {metro}")
    return value


def amenity_counts(elements, tract_ids, polygons):
    from shapely import STRtree, points
    records = {}
    for element in elements:
        key = (element["type"], element["id"])
        tags = element.get("tags", {})
        if tags.get("amenity") not in AMENITIES or tags.get("access") in {"no", "private", "members"}:
            continue
        if tags.get("disused") == "yes" or tags.get("abandoned") == "yes":
            continue
        location = element if element["type"] == "node" else element.get("center", {})
        if "lon" not in location or "lat" not in location:
            raise ValueError(f"Missing OSM feature coordinates: {key}")
        records[key] = (float(location["lon"]), float(location["lat"]))
    # The same OSM element may appear in overlapping download envelopes; deduplicate by type + ID.
    if not records:
        return pd.Series(0, index=tract_ids, name="amenities_count"), 0
    xy = np.asarray(list(records.values()))
    if not np.isfinite(xy).all() or not (np.abs(xy) <= [180, 90]).all():
        raise ValueError("Invalid OSM feature coordinates")
    pts = points(xy)
    pairs = STRtree(polygons).query(pts, predicate="intersects")
    # Boundary ties go to the lexicographically smallest GEOID, once only.
    assigned = pd.DataFrame({"point": pairs[0], "geoid": np.asarray(tract_ids)[pairs[1]]})
    assigned = assigned.sort_values("geoid").drop_duplicates("point")
    counts = assigned.groupby("geoid").size().reindex(tract_ids, fill_value=0).rename("amenities_count")
    return counts, len(records)


def enrich_study(root: Path):
    from shapely import make_valid
    from shapely.geometry import shape

    output = root / "data/processed/study"
    raw = root / "data/raw/study/environment"
    raw.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(output)
    frame = catalog.frame
    counties = sorted(frame.geoid.str[:5].unique())
    all_tracts, all_blocks = [], []
    print(f"Downloading tract polygons and populated 2020 block points for {len(counties)} counties", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(census_county, raw, c): c for c in counties}
        for i, job in enumerate(as_completed(jobs), 1):
            tract_features, blocks = job.result()
            all_tracts.extend(tract_features)
            all_blocks.append(blocks)
            if i % 10 == 0 or i == len(counties):
                print(f"Census counties: {i}/{len(counties)}", flush=True)
    tract_features = {f["properties"]["GEOID"]: f for f in all_tracts}
    if set(tract_features) != set(frame.geoid):
        raise ValueError("Downloaded tract polygons differ from the study inventory")
    polygons = [make_valid(shape(tract_features[g]["geometry"])) for g in frame.geoid]
    bounds = pd.DataFrame([p.bounds for p in polygons], columns=["west", "south", "east", "north"])
    bounds["metro"] = frame.metro.to_numpy()
    envelopes = bounds.groupby("metro").agg({"west": "min", "south": "min", "east": "max", "north": "max"})
    print("Checking ParkServe public-park IDs", flush=True)
    public = arcgis_inventory(raw, "public-parks", PARKS, "park_access=3", "parkid", page_size=2000)
    public_ids = set(public.parkid)
    def park_metro(metro, bbox):
        areas = features(raw, "service-" + metro, PARK_AREAS, bbox=bbox, fields="objectid,parkid")
        coverage = features(raw, "places-" + metro, PARK_PLACES, bbox=bbox, fields="objectid,id")
        return areas, coverage

    service, places, osm = {}, {}, {}
    osm_dates = []
    # Fetch independent ParkServe envelopes while running only one Overpass query at a time.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {m: pool.submit(park_metro, m, box.tolist()) for m, box in envelopes.iterrows()}
        for metro, box in envelopes.iterrows():
            print(f"ParkServe and OSM: metro {metro}", flush=True)
            areas, coverage = jobs.pop(metro).result()
            for f in areas:
                if f["properties"]["parkid"] in public_ids:
                    service[f["properties"]["objectid"]] = f
            for f in coverage:
                places[f["properties"]["objectid"]] = f
            result = osm_query(raw, metro, box.tolist())
            osm_dates.append(result["osm3s"]["timestamp_osm_base"])
            for element in result["elements"]:
                osm[(element["type"], element["id"])] = element
    if not service or not places or not osm:
        raise ValueError("Empty built-environment source download")
    print("Computing population-weighted block-point park access and mapping OSM features to tracts", flush=True)
    blocks = pd.concat(all_blocks, ignore_index=True)
    park = park_access(blocks, [make_valid(shape(f["geometry"])) for f in places.values()],
                       [make_valid(shape(f["geometry"])) for f in service.values()], frame)
    counts, mapped_total = amenity_counts(osm.values(), frame.geoid.to_numpy(), polygons)
    added = list(park.columns.difference(["geoid"])) + ["amenities_count", "amenities_per_1000"]
    frame = frame.drop(columns=added, errors="ignore").merge(park, on="geoid", validate="one_to_one")
    frame["amenities_count"] = frame.geoid.map(counts)
    frame["amenities_per_1000"] = 1000 * frame.amenities_count / frame.population.where(frame.population > 0)
    sources = [
        source_record("parkserve-service-snapshot", "Trust for Public Land ParkServe public parks and 10-minute walk service areas",
            "https://www.tpl.org/park-data-downloads", "Public service snapshot retrieved " + now(), "ParkServe polygons",
            [raw / f"service-{m}.geojson" for m in envelopes.index] + [raw / f"places-{m}.geojson" for m in envelopes.index] + [raw / "public-parks.json"],
            [PARK_DEFINITION, "ParkServe coverage is uneven outside its published urban places; incomplete population support remains missing.",
             "Block internal points approximate where residents live; access points and street networks have errors. Park access does not measure use, quality, safety, or social contact.",
             "Strict coverage selects a more urban/municipal subset; park-access analyses do not represent every tract in the metro.",
             "Service metadata includes legacy date labels; retrieval date is not an assertion that all parks were updated in that year.",
             "Trust for Public Land attribution and terms: https://parkserve.tpl.org/downloads/ParkServe_Terms_of_Use_May_2024.pdf"]),
        source_record("census-blocks-2020", "Census 2020 populated block counts and internal points", BLOCK20,
            "2020 Census; retrieved " + now(), "2020 Census blocks", [raw / f"blocks-{c}.json" for c in counties],
            ["2020 block populations weight park access; they are not current population estimates.",
             "Block parent tract IDs use the existing 2020–2023 inventory and land-area boundary review; identical IDs do not prove identical polygons."]),
        source_record("census-tract-polygons-2023", "Census TIGERweb ACS 2023 tract polygons", TRACT23,
            "2023 ACS tract boundaries; retrieved " + now(), "census2023-tract",
            [raw / f"tracts-{c}.geojson" for c in counties],
            ["Exact study tract inventory checked before assigning OSM coordinates to polygons."]),
        source_record("osm-gathering-places", "OpenStreetMap contributors: mapped gathering-place features", "https://www.openstreetmap.org/copyright",
            "OSM base dates " + min(osm_dates) + " to " + max(osm_dates), "Point/feature centers assigned to Census 2023 tracts",
            sorted(raw.glob("osm-*.json")), [OSM_DEFINITION,
             "© OpenStreetMap contributors, Open Database License (ODbL). Counted once per OSM type/ID; multiple mapped features can represent one venue.",
             "Zero means no matching mapped feature, not no real venue. Volunteer mapping completeness varies and commercial venues may require spending.",
             "Feature centers can fall outside the feature or on the other side of a tract boundary. Current mapped venues are temporally mismatched with older health and ACS estimates."]),
    ]
    manifest = dict(catalog.manifest)
    source_ids = {s["id"] for s in sources}
    manifest["sources"] = [s for s in manifest["sources"] if s["id"] not in source_ids] + sources
    for col in park.columns.difference(["geoid"]):
        manifest["column_sources"][col] = ["parkserve-service-snapshot", "census-blocks-2020", "census-tiger-2023"]
    manifest["column_sources"]["amenities_count"] = ["osm-gathering-places", "census-tract-polygons-2023"]
    manifest["column_sources"]["amenities_per_1000"] = ["osm-gathering-places", "acs5-2023-bulk", "census-tract-polygons-2023"]
    manifest["definitions"].update({"park_access_pct": PARK_DEFINITION, "amenities_per_1000": OSM_DEFINITION,
        "amenities_count": "Count of selected mapped OSM features assigned to the tract.",
        "park_population_2020": "2020 Census population summed over all populated blocks with this reviewed tract parent ID.",
        "park_covered_population_2020": "2020 block population whose internal point lies in published ParkServe place coverage.",
        "park_served_population_2020": "2020 block population whose internal point lies in both published coverage and an open-access park service area.",
        "park_population_coverage_pct": "Percent of 2020 block-point population inside published ParkServe place coverage; missing access values are not zeros."})
    manifest["caveats"] = [c for c in manifest["caveats"] if "park access and OSM amenities are not yet loaded" not in c]
    manifest["caveats"].extend(c for c in ["No direct loneliness or social-connection measures are loaded.",
        "Park access uses 2020 block population points and retrieved ParkServe polygons; OSM is a later mapped inventory. Neither measures actual use or social interaction."] if c not in manifest["caveats"])
    environment_coverage = frame.groupby("metro").agg(park_access=("park_access_pct", "count"),
        amenities=("amenities_per_1000", "count"), mapped_features=("amenities_count", "sum")).reset_index()
    coverage = pd.read_csv(output / "coverage.csv", dtype={"metro": str})
    coverage = coverage.drop(columns=["park_access", "amenities", "mapped_features"], errors="ignore").merge(environment_coverage, on="metro", validate="one_to_one")
    manifest["scope"]["coverage"] = coverage.to_dict(orient="records")
    manifest["snapshot"] = "top20-2023-environment-" + now()
    audit = json.loads((output / "audit.json").read_text())
    audit["environment"] = {"populated_blocks": len(blocks), "public_service_areas": len(service), "coverage_places": len(places),
        "osm_selected_features_in_envelopes": mapped_total, "osm_features_assigned_to_tracts": int(frame.amenities_count.sum()),
        "osm_base_dates": sorted(set(osm_dates)), "park_access_tracts": int(frame.park_access_pct.notna().sum()),
        "amenity_rate_tracts": int(frame.amenities_per_1000.notna().sum())}
    audit["missing_by_column"] = {c: int(frame[c].isna().sum()) for c in frame.columns}
    # Validate a candidate before replacing any served data.
    candidate = raw / "candidate"
    candidate.mkdir(exist_ok=True)
    frame.to_csv(candidate / "tracts.csv", index=False)
    coverage.to_csv(candidate / "coverage.csv", index=False)
    manifest["files"].update({name: digest(candidate / name) for name in ["tracts.csv", "coverage.csv"]})
    (candidate / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (candidate / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    Catalog(candidate)
    for name in ["tracts.csv", "coverage.csv", "audit.json", "manifest.json"]:
        (candidate / name).replace(output / name)
    print(json.dumps(audit["environment"], indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    raw = args.root / "data/raw/study"
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / ".build.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A data build is already running") from None
        enrich_study(args.root)
