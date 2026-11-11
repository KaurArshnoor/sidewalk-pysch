"""Read the real study, calculate statistics, and attach evidence to findings.

This is the data-access module used by app.py, analogous to wikipedia.py in the starter.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pydantic import BaseModel, ConfigDict, Field

# Shared findings format
DataRole = Literal["built_environment", "health_demographics", "analyst"]  # agents that run on the dataset
Role = Literal[DataRole, "literature"]  # research agents a user or the planner can run
Producer = Literal[Role, "reconciler"]  # everything that produces findings; reconciler = literature–data comparisons


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    url: str
    release: str
    geography: str
    retrieved_at: str
    sha256: str
    notes: list[str] = Field(default_factory=list)


class PaperSource(BaseModel):
    """A published paper or preprint that a literature finding cites."""

    model_config = ConfigDict(extra="forbid")
    id: str  # e.g. "doi:10.1016/j.healthplace.2020.102345" or "pmid:31234567"; matched by Finding.source_ids
    doi: str | None = None
    pmid: str | None = None
    title: str
    authors: list[str]
    year: int | None = None
    venue: str | None = None  # journal name, or "PsyArXiv" for preprints
    url: str
    database: Literal["pubmed", "openalex", "psyarxiv"]
    preprint: bool = False
    retrieved_at: str


class Effect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str  # e.g. "OR", "beta", "r", "SMD"
    value: float
    ci95: tuple[float, float] | None = None


class EvidenceStrength(BaseModel):
    """Computed in Python by the evidence-strength rubric, never by the model."""

    model_config = ConfigDict(extra="forbid")
    score: int
    tier: Literal["strong", "moderate", "weak"]
    reasons: list[str]


class StudyEvidence(BaseModel):
    """Evidence shape for literature findings; stored in Finding.evidence via model_dump()."""

    model_config = ConfigDict(extra="forbid")
    paper_id: str  # PaperSource.id
    design: Literal["meta_analysis", "rct", "longitudinal", "cross_sectional", "ecological", "qualitative", "other"]
    exposure: str  # what the study measured as the environment, e.g. "neighborhood walkability"
    outcome: str  # what it measured as the result, e.g. "depressive symptoms" (not necessarily loneliness)
    n: int | None = None
    effect: Effect | None = None
    preregistered: bool | None = None  # None when the paper does not say
    quote: str  # exact sentence from the fetched abstract that supports the extraction
    strength: EvidenceStrength | None = None


class Finding(BaseModel):
    """Shared handoff contract; all numerical findings are made by tools, not the LLM."""

    model_config = ConfigDict(extra="forbid")
    id: str
    agent: Producer
    claim: str
    evidence: dict
    source_ids: list[str]
    # descriptive/exploratory/insufficient: data agents; literature_*: literature agent, from the rubric tier
    confidence: Literal["descriptive", "exploratory", "insufficient",
                        "literature_strong", "literature_moderate", "literature_weak"]
    caveats: list[str]


class AgentDecision(BaseModel):
    finding_ids: list[str]
    gaps: list[str]
    clarification: str | None


class AgentResult(BaseModel):
    agent: Producer
    question: str
    status: Literal["ok", "needs_clarification", "insufficient"]
    findings: list[Finding]
    sources: list[Source | PaperSource]
    gaps: list[str] = Field(default_factory=list)
    clarification: str | None = None


class ToolCall(BaseModel):
    agent: str
    name: str
    arguments: dict
    result: dict
    duration_ms: float


# Dataset loading and source validation
ROOT = Path(__file__).resolve().parent
PERCENT_COLUMNS = {
    "distress_pct", "poverty_pct", "age65_pct", "black_pct", "hispanic_pct",
    "park_access_pct", "park_population_coverage_pct",
}
NONNEGATIVE_COLUMNS = {
    "population", "income", "pop_density", "amenities_per_1000", "amenities_count",
    "park_population_2020", "park_covered_population_2020", "park_served_population_2020",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate_tracts(frame: pd.DataFrame) -> None:
    if "geoid" not in frame or not frame.geoid.astype(str).str.fullmatch(r"\d{11}").all():
        raise ValueError("geoid must contain 11-digit tract FIPS strings, including leading zeros")
    if frame.geoid.duplicated().any():
        raise ValueError("Duplicate tract GEOIDs would multiply observations")


class Catalog:
    def __init__(self, directory: str | Path, *, allow_synthetic: bool = False):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.is_synthetic = self.manifest["synthetic"]
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("synthetic must be a boolean")
        if self.is_synthetic and not allow_synthetic:
            raise ValueError("Synthetic snapshots are not allowed in the app; load the real study dataset")
        csv = self.directory / "tracts.csv"
        if digest(csv) != self.manifest["files"]["tracts.csv"]:
            raise ValueError("Snapshot checksum mismatch: regenerate the manifest after rebuilding data")
        self.frame = pd.read_csv(csv, dtype={"geoid": str, "metro": str})
        validate_tracts(self.frame)
        if "metro" not in self.frame or self.frame.metro.isna().any():
            raise ValueError("Each tract needs an explicit metro membership")
        if "geography_id" not in self.manifest:
            raise ValueError("Snapshot needs a documented boundary geography_id")
        for col in self.frame.columns:
            if col in {"geoid", "metro"}:
                continue
            self.frame[col] = pd.to_numeric(self.frame[col], errors="raise")
            if np.isinf(self.frame[col].dropna()).any():
                raise ValueError(f"Infinite values in {col}")
            if col in PERCENT_COLUMNS and not self.frame[col].dropna().between(0, 100).all():
                raise ValueError(f"{col} must use 0–100 percentage units")
            if col in NONNEGATIVE_COLUMNS and (self.frame[col].dropna() < 0).any():
                raise ValueError(f"{col} must be nonnegative")
        if "walkability" in self.frame and not self.frame.walkability.dropna().between(1, 20).all():
            raise ValueError("walkability must lie in EPA's 1–20 range")
        self.sources = [Source.model_validate(s) for s in self.manifest["sources"]]
        self.source_ids = {s.id for s in self.sources}
        if len(self.source_ids) != len(self.sources):
            raise ValueError("Duplicate source IDs")
        for col in self.frame.columns.difference(["geoid", "metro"]):
            ids = self.manifest["column_sources"].get(col, [])
            if not ids or not set(ids) <= self.source_ids:
                raise ValueError(f"Missing provenance for {col}")

    @cached_property
    def adjacency(self) -> pd.DataFrame | None:
        expected = self.manifest["files"].get("tract-adjacency.csv")
        if expected is None:
            return None
        path = self.directory / "tract-adjacency.csv"
        if digest(path) != expected:
            raise ValueError("Tract adjacency checksum mismatch")
        edges = pd.read_csv(path, dtype=str)
        weights = self.manifest.get("spatial_weights", {})
        if weights.get("source_id") not in self.source_ids or weights.get("edges") != len(edges):
            raise ValueError("Tract adjacency provenance or edge count does not match the manifest")
        if list(edges.columns) != ["left", "right"] or edges.empty or edges.duplicated().any():
            raise ValueError("Invalid tract adjacency edge list")
        metros = self.frame.set_index("geoid").metro
        left, right = edges.left.map(metros), edges.right.map(metros)
        if left.isna().any() or not left.eq(right).all() or not edges.left.lt(edges.right).all():
            raise ValueError("Tract adjacency must connect distinct, known tracts within the same metro")
        return edges

    def subset(self, metro: str | None = None) -> pd.DataFrame:
        frame = self.frame if metro is None else self.frame[self.frame.metro == metro]
        if frame.empty:
            raise ValueError(f"No tracts available for metro {metro!r}")
        return frame.copy()

    def sources_for(self, columns: list[str]) -> list[str]:
        return sorted({s for c in columns for s in self.manifest["column_sources"].get(c, [])})

    def caveats(self) -> list[str]:
        caveats = list(self.manifest.get("caveats", []))
        if self.is_synthetic:
            caveats.insert(0, "SYNTHETIC TEST DATA: these values are invented and describe no real neighborhood.")
        return caveats

    def metadata(self) -> dict:
        return {
            "synthetic": self.is_synthetic, "snapshot": self.manifest["snapshot"],
            "geography_id": self.manifest["geography_id"],
            "rows": len(self.frame), "metros": sorted(self.frame.metro.unique()),
            "columns": self.manifest["column_sources"],
            "scope": self.manifest.get("scope", {}),
            "definitions": self.manifest.get("definitions", {}),
            "spatial_weights": self.manifest.get("spatial_weights"),
            "caveats": self.caveats(),
            "sources": [s.model_dump() for s in self.sources],
        }


# Statistical calculation
CONTROLS = ["poverty_pct", "income", "age65_pct", "black_pct", "hispanic_pct", "pop_density"]


SPATIAL_CAVEAT = ("Residual Moran’s I is descriptive. It uses tracts sharing a boundary or corner within a metro "
                  "and reports no significance test. Metro clustering is the covariance method when enough metros are available.")


def residual_moran(residuals: pd.Series, adjacency: pd.DataFrame | None) -> dict:
    """Binary, symmetric contiguity on the complete-case graph, excluding isolates."""
    if adjacency is None:
        return {"status": "not_run", "reason": "No tract adjacency file loaded"}
    left = residuals.index.get_indexer(adjacency.left)
    right = residuals.index.get_indexer(adjacency.right)
    retained = (left >= 0) & (right >= 0)
    left, right = left[retained], right[retained]
    linked = np.unique(np.concatenate([left, right]))
    if not linked.size:
        return {"status": "insufficient", "reason": "No adjacent complete-case tracts"}
    values = residuals.to_numpy()
    centered = values - values[linked].mean()
    variation = float(centered[linked] @ centered[linked])
    if variation <= np.finfo(float).eps:
        return {"status": "insufficient", "reason": "No meaningful residual variation among adjacent tracts"}
    return {"status": "ok", "moran_i": float(len(linked) * (centered[left] @ centered[right]) / (len(left) * variation)),
            "n_tracts": int(len(linked)), "n_isolated": int(len(values) - len(linked)), "n_edges": int(len(left))}


def association(frame: pd.DataFrame, adjusted: bool, moderator: str | None = None, *,
                adjacency: pd.DataFrame | None = None) -> dict:
    if moderator not in {None, "poverty_pct", "park_access_pct"}:
        raise ValueError("Supported moderators: poverty_pct or park_access_pct")
    columns = ["geoid", "metro", "walkability", "distress_pct"]
    if adjusted:
        columns += CONTROLS
    if moderator:
        columns.append(moderator)
    if adjacency is not None:
        columns.append("geoid")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        return {"status": "insufficient", "reason": f"Missing required columns: {missing}"}
    d = frame[list(dict.fromkeys(columns))].dropna().copy()
    base = {"n_available": len(frame), "n_complete": len(d), "n_excluded": len(frame) - len(d),
            "n_metros": int(d.metro.nunique()), "adjusted": adjusted, "moderator": moderator}
    if len(d) < 30 or d.walkability.nunique() < 3 or d.distress_pct.nunique() < 2:
        return {**base, "status": "insufficient", "reason": "Need at least 30 complete tracts and variation"}
    if adjusted:
        if (d.income <= 0).any() or (d.pop_density < 0).any():
            raise ValueError("Income must be positive and population density nonnegative")
        d["log_income"] = np.log(d.income)
        d["log_density"] = np.log1p(d.pop_density)
    terms = "walkability"
    if moderator:
        # One interaction unit equals ten percentage points; center before fitting.
        d["moderator_10pp"] = (d[moderator] - d[moderator].mean()) / 10
        terms += " * moderator_10pp"
    if adjusted:
        terms += " + poverty_pct + log_income + age65_pct + black_pct + hispanic_pct + log_density + C(metro)"
        if moderator == "poverty_pct":
            terms = terms.replace(" + poverty_pct", "")  # centered duplicate would be collinear
    formula = "distress_pct ~ " + terms
    model = smf.ols(formula, data=d)
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1] or len(d) <= model.exog.shape[1] + 5:
        return {**base, "status": "insufficient", "reason": "Singular or underdetermined design matrix"}
    # A cluster allows arbitrary residual dependence among tracts within that metro.
    # Suppress inferential quantities when there are too few independent metro clusters.
    groups = int(d.metro.nunique())
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": d.metro}, use_t=True) if groups >= 10 else model.fit()
    term = "walkability:moderator_10pp" if moderator else "walkability"
    ci = fitted.conf_int().loc[term].tolist() if groups >= 10 else None
    residuals = pd.Series(fitted.resid.to_numpy(), index=d.geoid) if adjacency is not None else fitted.resid
    return {
        **base, "status": "ok",
        "residual_spatial_diagnostic": residual_moran(residuals, adjacency), "formula": formula, "term": term,
        "estimate": float(fitted.params[term]), "ci95": ci,
        "p_value": float(fitted.pvalues[term]) if groups >= 10 else None,
        "pearson_r": float(d.walkability.corr(d.distress_pct)), "r_squared": float(fitted.rsquared),
        "standard_errors": "metro-clustered with finite-sample correction and t reference" if groups >= 10 else "inference suppressed: fewer than 10 metro clusters",
        "weighting": "equal weight per complete tract; not person-level",
        "moderator_mean_pct": float(d[moderator].mean()) if moderator else None,
    }


# Tools used by the agents
CAUSAL_LIMIT = "Cross-sectional tract associations cannot establish causation or individual effects."
MODELED_OUTCOME = "CDC PLACES is modeled from BRFSS and census/ACS inputs, not a direct survey of each tract."
CLUSTER_LIMIT = "Confidence intervals are suppressed for models with fewer than ten metro clusters."
ASSOCIATION_CAVEATS = [
    CAUSAL_LIMIT,
    MODELED_OUTCOME,
    "PLACES outcomes and ACS controls share demographic modeling inputs; adjusted patterns may partly reflect those models rather than independent tract health observations.",
    "PLACES model uncertainty and ACS margins of error are not propagated into regression intervals.",
    "Residential self-selection, reverse causality, and unmeasured confounding remain possible.",
    "Metro-clustered errors allow within-metro spatial dependence but assume independent metros; 20 clusters is still a small number.",
    "Analyses and interactions are exploratory; no multiple-testing adjustment is applied.",
    "Equal tract weighting does not estimate a population-average individual effect.",
]


@dataclass
class WorkContext:
    catalog: Catalog
    role: Role
    metro: str | None = None
    findings: dict[str, Finding] = field(default_factory=dict)
    calls: list[ToolCall] = field(default_factory=list)
    max_calls: int = 12

    @property
    def sources(self) -> list[Source]:
        return self.catalog.sources

    def finding(self, claim: str, evidence: dict, columns: list[str], caveats: list[str], confidence="descriptive", *, extra_sources: tuple[str, ...] = ()):
        source_ids = sorted(set(self.catalog.sources_for(columns)) | set(extra_sources))
        source_notes = [note for source in self.catalog.sources if source.id in source_ids for note in source.notes]
        body = {"agent": self.role, "claim": claim, "evidence": evidence,
                "source_ids": source_ids, "confidence": confidence,
                "caveats": list(dict.fromkeys(self.catalog.caveats() + source_notes + caveats))}
        identifier = "f-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        value = Finding(id=identifier, **body)
        self.findings[identifier] = value
        return value.model_dump()

    def invoke(self, name: str, **arguments) -> dict:
        if len(self.calls) >= self.max_calls:
            raise ValueError("Tool budget exhausted")
        start = time.perf_counter()
        try:
            if self.metro is not None and name in {"profile", "association"}:
                if arguments.get("metro") not in {None, self.metro}:
                    raise ValueError("Tool request conflicts with the selected metro")
                arguments["metro"] = self.metro
            result = self._invoke(name, **arguments)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            result = {"status": "error", "reason": str(exc)}
        self.calls.append(ToolCall(agent=self.role, name=name, arguments=arguments, result=result,
                                   duration_ms=round((time.perf_counter() - start) * 1000, 2)))
        return result

    def _invoke(self, name: str, **arguments) -> dict:
        if name == "dataset_metadata":
            metadata = self.catalog.metadata()
            coverage = {key: metadata[key] for key in ["snapshot", "synthetic", "geography_id", "rows", "metros", "scope"]}
            if self.metro is not None:
                selected = next((m for m in metadata["scope"].get("metro_lookup", []) if m["metro"] == self.metro), {})
                coverage.update(rows=len(self.catalog.subset(self.metro)), metros=[self.metro],
                    scope={"description": "Selected metro: " + selected.get("name", self.metro), "metro_count": 1})
            metadata["analysis_scope"] = coverage
            metadata["coverage_finding"] = self.finding(
                f"This analysis contains {coverage['rows']} tracts across {len(coverage['metros'])} metro areas "
                f"on {coverage['geography_id']} boundaries. Scope: {coverage['scope'].get('description', 'not documented')}.",
                coverage, list(self.catalog.manifest["column_sources"]),
                ["Coverage describes the loaded snapshot; it does not independently verify a top-20 metro universe."],
            )
            if self.role == "health_demographics" and {"cdc-places-2025", "acs5-2023-bulk"} <= self.catalog.source_ids:
                methods = {"cdc_places": MODELED_OUTCOME,
                           "acs": "The ACS values are period estimates from five years of survey data (2019–2023)."}
                metadata["health_measure_finding"] = self.finding(
                    " ".join(methods.values()), {"measurement_methods": methods},
                    ["distress_pct", "population"], ["The CDC and ACS measures describe different source periods."],
                )
            return metadata
        if name == "profile":
            allowed = {"built_environment": {"walkability", "amenities_per_1000", "amenities_count",
                                              "park_access_pct", "park_population_coverage_pct"},
                       "health_demographics": {"distress_pct", "population", "poverty_pct", "income", "age65_pct", "black_pct", "hispanic_pct"},
                       "analyst": set(self.catalog.frame.columns) - {"geoid", "metro"},
                       "literature": set()}[self.role]  # literature does not profile the dataset
            columns = arguments["columns"]
            if not columns or not set(columns) <= allowed:
                raise ValueError(f"Choose profile columns available to {self.role}: {sorted(allowed)}")
            d = self.catalog.subset(arguments.get("metro"))
            missing = sorted(set(columns) - set(d.columns))
            if missing:
                return {"status": "insufficient", "reason": f"Columns not loaded: {missing}"}
            stats = {}
            for col in columns:
                s = d[col].dropna()
                stats[col] = {"n": int(s.size), "missing": int(d[col].isna().sum()),
                              "mean": float(s.mean()) if s.size else None,
                              "median": float(s.median()) if s.size else None,
                              "min": float(s.min()) if s.size else None, "max": float(s.max()) if s.size else None}
            units = "Means are unweighted across tracts; percentages use 0–100 units; income is in dollars."
            parts = [f"{col}: mean {v['mean']:.3f} ({v['n']} nonmissing tracts)" if v["mean"] is not None
                     else f"{col}: no nonmissing observations" for col, v in stats.items()]
            return self.finding("; ".join(parts) + ".", {"n_tracts": len(d), "n_metros": int(d.metro.nunique()),
                                "metro_filter": arguments.get("metro"), "statistics": stats}, columns,
                                [units, "Frequent mental distress is not a direct measure of loneliness or social connection."] if "distress_pct" in columns else [units])
        if name == "association":
            if self.role != "analyst":
                raise ValueError("Only the analyst can fit models")
            d = self.catalog.subset(arguments.get("metro"))
            result = association(d, arguments.get("adjusted", True), arguments.get("moderator"),
                                 adjacency=self.catalog.adjacency)
            if result["status"] != "ok":
                return result
            result["metro_filter"] = arguments.get("metro")
            modifier = arguments.get("moderator")
            units = "percentage points of frequent mental distress per 1-point walkability increase"
            if modifier:
                units += f", slope change per 10 percentage points of {modifier}"
            claim = (f"The {'adjusted' if result['adjusted'] else 'unadjusted'} tract-level association is "
                     f"{result['estimate']:.4f} {units} (n={result['n_complete']}; {result['n_metros']} metros).")
            spatial = result["residual_spatial_diagnostic"]
            if spatial["status"] == "ok":
                claim += f" Residual Moran’s I is {spatial['moran_i']:.3f} across {spatial['n_tracts']:,} tracts with neighbors."
            columns = ["walkability", "distress_pct"]
            if result["adjusted"]:
                columns += CONTROLS
            if modifier:
                columns += [modifier]
            caveats = ASSOCIATION_CAVEATS + ([CLUSTER_LIMIT] if result["ci95"] is None else [])
            spatial_source = self.catalog.manifest.get("spatial_weights", {}).get("source_id")
            used_spatial = spatial["status"] == "ok"
            return self.finding(claim, result, columns, caveats + ([SPATIAL_CAVEAT] if used_spatial else []),
                                confidence="exploratory", extra_sources=(spatial_source,) if used_spatial and spatial_source else ())
        raise ValueError(f"Unknown tool: {name}")
