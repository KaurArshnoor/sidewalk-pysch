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
