import json

import httpx
import pytest

from app import assemble, run_agent
from data import ROOT, Catalog
from data import AgentDecision
from data import WorkContext


@pytest.mark.parametrize("role", ["built_environment", "health_demographics", "analyst"])
def test_independent_findings_contract(role, catalog):
    ctx = WorkContext(catalog, role)
    ctx.invoke("dataset_metadata")
    columns = ["distress_pct"] if role == "health_demographics" else ["walkability"]
    ctx.invoke("profile", columns=columns)
    decision = AgentDecision(finding_ids=list(ctx.findings), gaps=[], clarification=None)
    result = assemble(ctx, "Summarize the requested measure", decision)
    assert result.findings
    assert all(c.agent == role for c in ctx.calls)
    assert all(set(f.source_ids) <= {s.id for s in result.sources} for f in result.findings)
    assert all(f.claim and f.evidence and f.caveats for f in result.findings)


def test_model_cannot_invent_finding_ids(catalog):
    with pytest.raises(ValueError, match="did not obtain"):
        assemble(WorkContext(catalog, "analyst"), "question", AgentDecision(finding_ids=["invented"], gaps=[], clarification=None))


def test_role_cannot_fit_models(catalog):
    assert WorkContext(catalog, "built_environment").invoke("association", adjusted=True)["status"] == "error"


def test_unavailable_measures_are_explicit(catalog):
    assert WorkContext(catalog, "health_demographics").invoke("profile", columns=["loneliness"])["status"] == "error"


def test_source_tampering_detected(tmp_path, catalog):
    import shutil
    shutil.copytree(catalog.directory, tmp_path / "copy")
    with (tmp_path / "copy/tracts.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        Catalog(tmp_path / "copy", allow_synthetic=True)


def test_app_rejects_synthetic_snapshots(catalog):
    with pytest.raises(ValueError, match="Synthetic snapshots are not allowed"):
        Catalog(catalog.directory)


async def test_agent_rejects_synthetic_snapshots(catalog):
    with pytest.raises(ValueError, match="Synthetic snapshots are not allowed"):
        await run_agent("analyst", "Compare the two association models", catalog.directory)


@pytest.fixture
def api_app(monkeypatch):
    import app as module
    monkeypatch.setenv("SIDEWALK_DATASET", str(ROOT / "data/processed/study"))

    async def stub_agent(role, question, dataset, *, hooks=None):
        # Test HTTP transport and real data access without purchasing a model call.
        ctx = WorkContext(Catalog(dataset), role)
        ctx.invoke("dataset_metadata")
        ctx.invoke("profile", columns=["distress_pct"] if role == "health_demographics" else ["walkability"])
        decision = AgentDecision(finding_ids=list(ctx.findings), gaps=[], clarification=None)
        return assemble(ctx, question, decision), ctx.calls

    monkeypatch.setattr(module, "run_agent", stub_agent)
    return module.app


async def test_api_shape_and_agent_trace(api_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "Summarize walkability", "agent": "analyst"})
        assert response.status_code == 200
        body = response.json()
        assert {"response", "tool_calls", "result", "run_id", "mode"} <= set(body)
        assert body["mode"] == "live"
        assert all("agent" in call for call in body["tool_calls"])
        assert not any("SYNTHETIC TEST DATA" in " ".join(f["caveats"]) for f in body["result"]["findings"])


async def test_sse_finishes_and_matches_chat(api_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://test") as client:
        response = await client.get("/run", params={"question": "Summarize walkability", "agent": "built_environment"})
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[0]["type"] == "start"
        assert events[-1]["type"] == "done"


async def test_invalid_agent_rejected(api_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "Summarize walkability", "agent": "writer"})
        assert response.status_code == 422


async def test_missing_snapshot_fails_without_fallback(api_app, monkeypatch, tmp_path):
    monkeypatch.setenv("SIDEWALK_DATASET", str(tmp_path / "missing"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "Summarize walkability"})
        assert response.status_code == 503


async def test_unconfigured_dataset_defaults_to_real_study(api_app, monkeypatch):
    monkeypatch.delenv("SIDEWALK_DATASET", raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["synthetic"] is False
        assert response.json()["rows"] == 30056
