"""Spatial ETL checks; small artificial geometries test methods, not study findings."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")
from shapely.geometry import box  # noqa: E402

from scripts.build_environment import amenity_counts, features, park_access  # noqa: E402


def test_park_population_weighting_overlap_and_missing_coverage():
    blocks = pd.DataFrame({
        "GEOID": ["010010001001001", "010010001001002", "010010002001001", "010010003001001"],
        "POP100": [90, 10, 40, 30], "INTPTLON": [.5, 1.5, 3.5, .5], "INTPTLAT": [.5] * 4,
    })
    tracts = pd.DataFrame({"geoid": ["01001000100", "01001000200", "01001000300", "01001000400"],
                           "boundary_review_excluded": [0, 0, 1, 0]})
    result = park_access(blocks, [box(0, 0, 2, 2)], [box(0, 0, 1, 1), box(0, 0, 1.2, 1)], tracts).set_index("geoid")
    assert result.loc["01001000100", "park_access_pct"] == 90  # population, not 50% of blocks
    assert result.loc["01001000100", "park_population_coverage_pct"] == 100
    assert result.loc["01001000200", "park_population_coverage_pct"] == 0
    assert result.loc[["01001000200", "01001000300", "01001000400"], "park_access_pct"].isna().all()


def test_park_zero_is_valid_only_for_fully_covered_tract():
    blocks = pd.DataFrame({"GEOID": ["010010001001001"], "POP100": [10], "INTPTLON": [.5], "INTPTLAT": [.5]})
    tracts = pd.DataFrame({"geoid": ["01001000100"], "boundary_review_excluded": [0]})
    result = park_access(blocks, [box(0, 0, 1, 1)], [], tracts)
    assert result.park_access_pct.iloc[0] == 0


def test_osm_identity_dedup_access_filters_and_boundary_assignment():
    def node(identifier, x, access=None):
        tags = {"amenity": "cafe"}
        if access:
            tags["access"] = access
        return {"type": "node", "id": identifier, "lon": x, "lat": .5, "tags": tags}
    elements = [node(1, .5), node(1, .5), node(2, 1), node(3, 1.5, "private"), node(4, 1.5, "members"),
                {"type": "way", "id": 1, "center": {"lon": 1.5, "lat": .5}, "tags": {"amenity": "library"}}]
    counts, selected = amenity_counts(elements, np.array(["a", "b"]), [box(0, 0, 1, 1), box(1, 0, 2, 1)])
    assert counts.to_dict() == {"a": 2, "b": 1}
    assert selected == 3


def test_arcgis_geometry_truncation_does_not_create_a_completed_cache(tmp_path, monkeypatch):
    def fake(raw, name, url, params):
        if params.get("returnIdsOnly"):
            return {"objectIds": [1, 2], "objectIdFieldName": "OBJECTID"}
        return {"features": [{"properties": {"OBJECTID": 1}, "geometry": {"type": "Point", "coordinates": [0, 0]}}]}
    monkeypatch.setattr("scripts.build_environment.cache_json", fake)
    with pytest.raises(ValueError, match="Incomplete geometry"):
        features(tmp_path, "test", "https://example.org")
    assert not (tmp_path / "test.geojson").exists()


def test_partial_overpass_response_never_becomes_a_zero_count(tmp_path, monkeypatch):
    from scripts.build_environment import osm_query
    import httpx
    def incomplete(*args, **kwargs):
        return httpx.Response(200, json={"elements": [], "remark": "runtime error: query timed out"},
                              request=httpx.Request("GET", "https://example.org"))
    monkeypatch.setattr("scripts.build_environment.httpx.get", incomplete)
    monkeypatch.setattr("scripts.build_environment.time.sleep", lambda _: None)
    with pytest.raises(ValueError, match="no zeros were written"):
        osm_query(tmp_path, "12345", [0, 0, 1, 1])
    assert not (tmp_path / "osm-12345.json").exists()
