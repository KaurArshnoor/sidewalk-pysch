"""SDK wiring tests: no API calls or provider credentials required."""
import pytest

pytest.importorskip("agents")

from agents import RunContextWrapper  # noqa: E402

from app import create_agent, run_agent  # noqa: E402
from data import WorkContext
from data import ROOT  # noqa: E402


async def test_metadata_first_and_profile_budget(monkeypatch, catalog):
    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    agent = create_agent("built_environment")
    ctx = RunContextWrapper(context=WorkContext(catalog, "built_environment"))
    metadata, profile = agent.tools
    assert metadata.is_enabled(ctx, agent)
    assert not profile.is_enabled(ctx, agent)
    ctx.context.invoke("dataset_metadata")
    assert not metadata.is_enabled(ctx, agent)
    assert profile.is_enabled(ctx, agent)
    for _ in range(2):
        ctx.context.invoke("profile", columns=["walkability"])
    assert not profile.is_enabled(ctx, agent)


def test_role_tools_and_fresh_context(monkeypatch, catalog):
    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    assert [t.name for t in create_agent("analyst").tools] == ["dataset_metadata", "profile", "fit_association"]
    a, b = WorkContext(catalog, "analyst"), WorkContext(catalog, "analyst")
    a.invoke("dataset_metadata")
    assert not b.calls


async def test_generic_request_clarifies_without_a_model(catalog, monkeypatch):
    monkeypatch.delenv("SIDEWALK_MODEL", raising=False)
    result, calls = await run_agent("analyst", "Please explore our dataset", ROOT / "data/processed/study")
    assert result.status == "needs_clarification"
    assert result.clarification and not result.findings
    assert calls[0].name == "dataset_metadata"
