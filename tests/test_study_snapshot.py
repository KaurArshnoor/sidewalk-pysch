from pathlib import Path

import pytest

from data import ROOT, Catalog, digest

STUDY = ROOT / "data/processed/study"
pytestmark = pytest.mark.skipif(not (STUDY / "manifest.json").exists(), reason="Prepared study snapshot not present")


def test_shipped_study_is_real_and_contains_twenty_complete_metros():
    c = Catalog(STUDY)
    assert c.is_synthetic is False and c.manifest["scope"]["top20_verified"] is True
    assert c.frame.metro.nunique() == 20 and len(c.frame) > 10000
    assert all(row["complete_adjusted"] > 0 for row in c.manifest["scope"]["coverage"])
    assert len(c.manifest["scope"]["metro_lookup"]) == 20


def test_all_shipped_artifact_checksums_match():
    c = Catalog(STUDY)
    for name, expected in c.manifest["files"].items():
        assert digest(Path(STUDY, name)) == expected


def test_primary_exposure_respects_support_and_boundary_checks():
    d = Catalog(STUDY).frame
    primary = d[d.walkability.notna()]
    assert primary.epa_population_coverage_pct.ge(100 - 1e-9).all()
    assert primary.boundary_review_excluded.eq(0).all()
    assert d.loc[d.boundary_review_excluded.eq(1), "walkability"].isna().all()
    assert d.income.isna().any() and d.distress_pct.isna().any()


def test_structural_health_gap_is_disclosed():
    c = Catalog(STUDY)
    missing = c.frame[c.frame.geoid.str.startswith("42")]
    assert not missing.empty and missing.distress_pct.isna().all()
    assert any(g["state_fips"] == "42" for g in c.manifest["scope"]["structural_health_gaps"])
    assert "Pennsylvania" in " ".join(c.caveats())


def test_real_environment_layers_and_coverage_are_present():
    import numpy as np
    c = Catalog(STUDY)
    d = c.frame
    assert {"park_access_pct", "park_population_coverage_pct", "amenities_count", "amenities_per_1000"} <= set(d)
    park = d[d.park_access_pct.notna()]
    assert park.metro.nunique() == 20
    assert park.park_population_coverage_pct.eq(100).all()
    assert park.boundary_review_excluded.eq(0).all()
    assert d.loc[d.park_population_coverage_pct.lt(100), "park_access_pct"].isna().all()
    assert d.amenities_count.notna().all() and d.amenities_count.ge(0).all()
    assert d.amenities_count.eq(0).any() and d.amenities_count.gt(0).any()
    populated = d.population.gt(0)
    assert np.allclose(d.loc[populated, "amenities_per_1000"],
                       1000 * d.loc[populated, "amenities_count"] / d.loc[populated, "population"])
    assert d.loc[~populated, "amenities_per_1000"].isna().all()
    assert {"parkserve-service-snapshot", "osm-gathering-places", "census-blocks-2020", "census-tract-polygons-2023"} <= c.source_ids
    assert all(row["park_access"] > 0 and row["amenities"] > 0 for row in c.manifest["scope"]["coverage"])
