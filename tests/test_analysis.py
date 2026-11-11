import numpy as np
import pandas as pd
import pytest

from data import association, residual_moran


def test_recovers_known_association(catalog):
    d = catalog.frame.copy()
    # Exactly known linear signal plus noise independently generated from covariates.
    rng = np.random.default_rng(17)
    d["distress_pct"] = 20 - 0.4 * d.walkability + rng.normal(0, 0.05, len(d))
    result = association(d, adjusted=True)
    assert result["status"] == "ok"
    assert result["estimate"] == pytest.approx(-0.4, abs=0.003)
    assert result["ci95"][0] < -0.4 < result["ci95"][1]
    assert result["n_metros"] == 20


def test_missing_controls_never_silently_dropped(catalog):
    result = association(catalog.frame.drop(columns="income"), adjusted=True)
    assert result["status"] == "insufficient" and "income" in result["reason"]


def test_complete_case_counts(catalog):
    d = catalog.frame.copy()
    d.loc[:7, "poverty_pct"] = np.nan
    result = association(d, adjusted=True)
    assert result["n_excluded"] == 8
    assert result["n_complete"] == len(d) - 8


def test_few_clusters_suppress_intervals(catalog):
    d = catalog.frame.iloc[:48]
    result = association(d, adjusted=False)
    assert result["status"] == "ok"
    assert result["ci95"] is None and result["p_value"] is None


def test_no_variation_is_insufficient(catalog):
    d = catalog.frame.assign(walkability=10)
    assert association(d, adjusted=False)["status"] == "insufficient"


def test_collinearity_is_insufficient(catalog):
    d = catalog.frame.assign(age65_pct=catalog.frame.poverty_pct)
    assert association(d, adjusted=True)["status"] == "insufficient"


@pytest.mark.parametrize("moderator", ["poverty_pct", "park_access_pct"])
def test_interaction_units(catalog, moderator):
    d = catalog.frame.copy()
    centered = (d[moderator] - d[moderator].mean()) / 10
    d["distress_pct"] = 20 - 0.2 * d.walkability + 0.07 * d.walkability * centered
    result = association(d, adjusted=True, moderator=moderator)
    assert result["status"] == "ok"
    assert result["estimate"] == pytest.approx(0.07, abs=1e-6)


def test_unknown_moderator_rejected(catalog):
    with pytest.raises(ValueError):
        association(catalog.frame, adjusted=True, moderator="loneliness")



def test_moran_distinguishes_grouped_residuals_and_excludes_missing_neighbors():
    edges = pd.DataFrame({"left": ["a", "b", "c"], "right": ["b", "missing", "d"]})
    grouped = pd.Series([-1.0, -1.0, 1.0, 1.0, 99.0], index=["a", "b", "c", "d", "isolated"])
    alternating = pd.Series([-1.0, 1.0, -1.0, 1.0, 99.0], index=grouped.index)
    result = residual_moran(grouped, edges)
    assert result["moran_i"] == pytest.approx(1.0)
    assert (result["n_edges"], result["n_tracts"], result["n_isolated"]) == (2, 4, 1)
    assert residual_moran(alternating, edges)["moran_i"] == pytest.approx(-1.0)
