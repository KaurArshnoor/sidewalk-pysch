import csv
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.build_data import ACS_VARIABLES
from scripts.build_data import arcgis_inventory, crosswalk_weights, parse_crosswalk_row, read_acs_bulk, review_boundaries, select_metros


def test_rank_metro_totals_and_restore_county_fips():
    records = []
    for index in range(21):
        code = str(10000 + index)
        records += [dict(CBSA=code, LSAD="Metropolitan Statistical Area", NAME=f"Metro {index}",
                         POPESTIMATE2023=100000 - index, STCOU=None),
                    dict(CBSA=code, LSAD="County or equivalent", NAME=f"County {index}",
                         POPESTIMATE2023=100000 - index, STCOU=str(1001 + index))]
    records.append(dict(CBSA="99999", LSAD="Micropolitan Statistical Area", NAME="Exclude me",
                        POPESTIMATE2023=9999999, STCOU=None))
    metros, counties = select_metros(pd.DataFrame(records))
    assert len(metros) == 20 and len(counties) == 20
    assert metros.metro.iloc[0] == "10000" and "99999" not in set(metros.metro)
    assert counties.county.iloc[0] == "01001"


def test_bulk_acs_same_cell_mapping_and_exact_geography(tmp_path):
    values = {code: "1" for code in ACS_VARIABLES}
    values.update({"B01003_001E": "100", "B17001_001E": "80", "B17001_002E": "20",
                   "B19013_001E": "-666666666"})
    for table in {code.split("_")[0] for code in values}:
        variables = [code for code in values if code.startswith(table + "_")]
        headers = [f"{table}_{code[-1]}{code.split('_')[1][:-1]}" for code in variables]
        with (tmp_path / f"acs-{table.lower()}.dat").open("w") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["GEO_ID", *headers])
            for identifier in ["1400000US01001000100", "1400000US01003000100", "1400001US01001000100", "0100000US"]:
                writer.writerow([identifier, *[values[v] for v in variables]])
    frame = read_acs_bulk(tmp_path, {"01001"})
    assert len(frame) == 1 and frame.geoid.iloc[0] == "01001000100"
    assert frame.poverty_pct.iloc[0] == 25
    assert frame.age65_pct.iloc[0] == 12
    assert pd.isna(frame.income.iloc[0])


@pytest.mark.parametrize("row", [["010010003001001", "010010001001"],
                                  ["010010003001001", "010010001001", "nan"],
                                  ["010010003001001", "010010001001", "1.01"]])
def test_crosswalk_parser_rejects_invalid_shares(row):
    with pytest.raises(ValueError):
        parse_crosswalk_row(row)


def crosswalk_fixture(tmp_path, missing_score=False):
    a, b = "010010001001", "010010002001"
    # Second target is outside study scope. It MUST still count in source normalization.
    content = f"010010003001001,{a},0.25,{b},0.5\n010030004001001,{a},0.75,{b},0.5\n"
    with zipfile.ZipFile(tmp_path / "uf-bg19-block20.zip", "w") as z:
        z.writestr("bg19block20_pop_crosswalk_01.csv", content)
    with zipfile.ZipFile(tmp_path / "uf-acs19-block20-pop.zip", "w") as z:
        z.writestr("pop19_01.csv", "010010003001001,175\n010030004001001,225\n")
    return pd.DataFrame({"GEOID20": [a, b], "TotPop": [100, 300],
                         "NatWalkInd": [2, np.nan if missing_score else 10]})


def test_crosswalk_multiplies_shares_by_source_population(tmp_path):
    epa = crosswalk_fixture(tmp_path)
    result, weights, audit = crosswalk_weights(tmp_path, epa, {"01001"})
    assert result.walkability.iloc[0] == pytest.approx((25 * 2 + 150 * 10) / 175)
    assert weights.population_weight.sum() == 175
    assert audit["max_outgoing_share_deviation"] == 0
    assert result.epa_population_coverage_pct.iloc[0] == 100


def test_crosswalk_missing_support_is_not_silently_renormalized(tmp_path):
    epa = crosswalk_fixture(tmp_path, missing_score=True)
    result, _, _ = crosswalk_weights(tmp_path, epa, {"01001"})
    assert pd.isna(result.walkability.iloc[0])
    assert result.walkability_partial.iloc[0] == 2
    assert result.epa_population_coverage_pct.iloc[0] == 0


def test_arcgis_pagination_checks_count_and_caches(monkeypatch, tmp_path):
    def fake(raw, name, url, params):
        if params.get("returnCountOnly"):
            return {"count": 3}
        offset = params["resultOffset"]
        return {"features": [{"attributes": {"id": i}} for i in range(offset, min(offset + 2, 3))]}
    monkeypatch.setattr("scripts.build_data.cache_json", fake)
    frame = arcgis_inventory(tmp_path, "test", "https://example.org", "1=1", "id", page_size=2)
    assert list(frame.id) == [0, 1, 2]
    saved = json.loads((tmp_path / "test.json").read_text())
    assert saved["expected_count"] == 3


def test_arcgis_truncation_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.build_data.cache_json", lambda *a: {"count": 3} if a[-1].get("returnCountOnly") else {"features": []})
    with pytest.raises(ValueError, match="Truncated"):
        arcgis_inventory(tmp_path, "test", "https://example.org", "1=1", "id")


def test_boundary_review_flags_large_changes_and_undefined_areas():
    old = pd.DataFrame({"geoid": ["a", "b", "c"], "land_area_2020": [100, 100, 0]})
    new = pd.DataFrame({"geoid": ["a", "b", "c"], "land_area_2023": [101, 102, 10]})
    result = review_boundaries(new, old)
    assert list(result.boundary_review_excluded) == [0, 1, 1]
    with pytest.raises(ValueError, match="IDs changed"):
        review_boundaries(new.iloc[:2], old)
