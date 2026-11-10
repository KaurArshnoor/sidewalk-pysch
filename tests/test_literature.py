"""Literature agent: query expansion, parsers, extraction checks and rubric on recorded real responses.

No network or model calls. Refresh responses with: uv run python -m scripts.record_literature_responses
"""
import json
import re
from types import SimpleNamespace

import httpx
import pytest

import literature
from data import ROOT, AgentDecision, AgentResult, PaperSource, StudyEvidence
from literature import LiteratureContext, expand, parse_osf, preregistered, pubmed_query, score
from scripts.record_literature_responses import SEARCHES

RECORDED = json.loads((ROOT / "tests/literature_responses.json").read_text())


def transport(status=None):
    def handle(request: httpx.Request):
        if status is not None:
            return httpx.Response(status)
        response = RECORDED.get(str(request.url))
        if response is None:
            return httpx.Response(500, text="not recorded")
        if response["json"] is not None:
            return httpx.Response(response["status"], json=response["json"])
        return httpx.Response(response["status"], text=response["text"])
    return httpx.MockTransport(handle)


@pytest.fixture(autouse=True)
def no_openalex_key(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)  # recorded URLs have no key


@pytest.fixture
async def ctx():
    async with httpx.AsyncClient(transport=transport()) as client:
        context = LiteratureContext(client=client)
        context.results = {name: await context.invoke(name, concepts=concepts) for name, concepts in SEARCHES}
        yield context


def with_number(context: LiteratureContext, database=None):
    """A searched paper, one abstract sentence from it, and an integer of at least 20 in that sentence."""
    for paper in context.papers.values():
        if database and paper.database != database:
            continue
        for sentence in re.split(r"(?<=\.)\s+", paper.abstract):
            for raw in re.findall(r"(?<![\d.,])\d[\d,]*(?![\d.])", sentence):
                value = int(raw.replace(",", ""))
                if value >= 20 and not 1900 <= value <= 2100 and len(sentence) >= 40:
                    return paper, sentence, value
    raise AssertionError("no abstract sentence with a sample-size-like number")


def without_randomization(context: LiteratureContext):
    return next(p for p in context.papers.values()
                if not re.search(r"randomi[sz]ed|randomly", p.abstract + p.title + " ".join(p.publication_types), re.I))


# Queries

def test_expansion_and_query_syntax():
    groups = expand(["Walkability", "loneliness", 'green, "space" (urban)', "fourth concept"])
    assert len(groups) == 3  # at most three concepts
    assert "walkable neighborhood" in groups[0] and "social isolation" in groups[1]
    assert groups[2] == ["green space urban"]  # unknown concept passes through without quotes or commas
    query = pubmed_query(groups[:2])
    assert query.startswith('("walkability"[tiab] OR ') and ') AND ("loneliness"[tiab]' in query


# Searches on recorded responses

async def test_searches_parse_real_responses(ctx):
    results = ctx.results
    assert all(r["status"] == "ok" for r in results.values())
    pubmed, openalex, psyarxiv = (results[name]["papers"] for name, _ in SEARCHES)
    assert pubmed and openalex and psyarxiv
    assert all(p["abstract"] and p["title"] for p in pubmed + openalex + psyarxiv)
    assert all(ctx.papers[p["paper_id"]].pmid for p in pubmed)
    assert all(p["preprint"] and p["venue"] == "PsyArXiv" for p in psyarxiv)
    assert "walkability" in results["search_pubmed"]["query"]  # the exact query is shown to the model
    bases = [re.sub(r"_v\d+$", "", p["paper_id"]) for p in psyarxiv]
    assert len(bases) == len(set(bases))  # one version per preprint


async def test_openalex_abstract_is_rebuilt_in_word_order(ctx):
    data = next(r["json"] for url, r in RECORDED.items() if "openalex" in url)
    work = next(w for w in data["results"] if w.get("abstract_inverted_index"))
    first_word = min(work["abstract_inverted_index"].items(), key=lambda item: min(item[1]))[0]
    paper = next(p for p in ctx.papers.values() if p.database == "openalex" and p.title == work["title"])
    assert paper.abstract.startswith(first_word)
    assert len(paper.abstract.split()) == sum(len(v) for v in work["abstract_inverted_index"].values())


async def test_repeated_search_returns_only_new_papers(ctx):
    name, concepts = SEARCHES[0]
    again = await ctx.invoke(name, concepts=concepts)
    assert again["papers"] == [] and again["already_seen"] == len(ctx.results[name]["papers"])


async def test_unavailable_database_is_a_gap_not_a_crash(monkeypatch):
    async def no_wait(seconds):
        pass
    monkeypatch.setattr(literature.apis.asyncio, "sleep", no_wait)
    async with httpx.AsyncClient(transport=transport(status=429)) as client:
        result = await LiteratureContext(client=client).invoke("search_openalex", concepts=["walkability"])
    assert result["status"] == "unavailable" and "gap" in result["note"]


# record_study

async def test_record_study_creates_a_checked_finding(ctx):
    paper, sentence, n = with_number(ctx)
    finding = await ctx.invoke("record_study", paper_id=paper.id, design="cross_sectional", quote=sentence,
                               exposure="neighborhood walkability", outcome="depressive symptoms", n=n)
    assert finding["agent"] == "literature" and finding["confidence"].startswith("literature_")
    evidence = StudyEvidence.model_validate(finding["evidence"])
    assert evidence.n == n and evidence.strength.reasons[0] == "cross-sectional design +1"
    assert f"n={n:,}" in finding["claim"] and sentence.split()[0] in finding["claim"]
    assert any("not loneliness" in c for c in finding["caveats"])
    assert [s.id for s in ctx.sources] == [paper.id] and isinstance(ctx.sources[0], PaperSource)
    again = await ctx.invoke("record_study", paper_id=paper.id, design="cross_sectional", quote=sentence,
                             exposure="x", outcome="y")
    assert again["status"] == "error" and "already recorded" in again["reason"]


async def test_record_study_rejects_unchecked_extractions(ctx):
    paper, sentence, n = with_number(ctx)
    base = dict(paper_id=paper.id, design="cross_sectional", quote=sentence, exposure="walkability", outcome="mood")
    cases = {
        "Unknown paper_id": dict(base, paper_id="doi:10.9999/invented"),
        "copied exactly": dict(base, quote=sentence.replace(" ", "  ") + " This was not in the abstract."),
        "does not appear in the quote": dict(base, n=n + 1),
        "effect_metric is required": dict(base, effect_value=float(n)),
        "exposure and outcome": dict(base, outcome="  "),
    }
    for expected, arguments in cases.items():
        result = await ctx.invoke("record_study", **arguments)
        assert result["status"] == "error" and expected in result["reason"], (expected, result)
    observational = without_randomization(ctx)
    claim_rct = dict(base, paper_id=observational.id, quote=observational.abstract[:80].rsplit(" ", 1)[0], design="rct")
    result = await ctx.invoke("record_study", **claim_rct)
    assert "does not describe a randomized trial" in result["reason"]
    assert ctx.findings == {}


async def test_paper_id_may_be_a_bare_doi_or_pmid(ctx):
    paper = next(p for p in ctx.papers.values() if p.doi and p.pmid)
    quote = paper.abstract[:60].rsplit(" ", 1)[0]
    finding = await ctx.invoke("record_study", paper_id=paper.pmid, design="other", quote=quote,
                               exposure="built environment", outcome="mental health")
    assert finding["source_ids"] == [paper.id]


# Rubric and preregistration

@pytest.mark.parametrize("design,n,prereg,preprint,points,tier", [
    ("meta_analysis", 12000, True, False, 5, "strong"),
    ("longitudinal", 4120, None, False, 3, "moderate"),
    ("cross_sectional", 500, None, False, 1, "weak"),
    ("rct", 200, True, True, 3, "moderate"),
    ("ecological", None, None, True, -1, "weak"),
])
def test_rubric(design, n, prereg, preprint, points, tier):
    strength = score(design, n, prereg, preprint)
    assert (strength.score, strength.tier) == (points, tier) and strength.reasons


def test_preregistration_detection():
    assert preregistered("The protocol was registered in PROSPERO (CRD42020123456).") is True
    assert preregistered("Trial registration: NCT01234567.") is True
    assert preregistered("This analysis was not preregistered.") is None  # unknown, never inferred False
    item = {"id": "abc12_v2", "links": {"preprint_doi": "https://doi.org/10.31234/osf.io/abc12_v2"},
            "attributes": {"title": "T", "description": "An abstract.", "has_prereg_links": "no"}}
    assert parse_osf({"data": [item]})[0].preregistered is False
    item["attributes"]["has_prereg_links"] = "available"
    assert parse_osf({"data": [item]})[0].preregistered is True


# Agent wiring

def test_literature_agent_tools_and_budgets(monkeypatch):
    pytest.importorskip("agents")
    from agents import RunContextWrapper
    from app import create_agent
    from data import ToolCall

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    agent = create_agent("literature")
    assert [t.name for t in agent.tools] == ["search_pubmed", "search_openalex", "search_psyarxiv", "record_study"]
    wrapper = RunContextWrapper(context=LiteratureContext())
    pubmed = agent.tools[0]
    assert pubmed.is_enabled(wrapper, agent)
    for _ in range(2):
        wrapper.context.calls.append(ToolCall(agent="literature", name="search_pubmed", arguments={}, result={},
                                              duration_ms=0))
    assert not pubmed.is_enabled(wrapper, agent)


async def test_run_agent_returns_paper_sources(monkeypatch):
    pytest.importorskip("agents")
    import agents
    from app import run_agent

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")

    async def fake_run(agent, prompt, *, context, **kwargs):  # stands in for the model's tool use
        await context.invoke(*SEARCHES[0][:1], concepts=SEARCHES[0][1])
        paper, sentence, n = with_number(context)
        finding = await context.invoke("record_study", paper_id=paper.id, design="cross_sectional",
                                       quote=sentence, exposure="walkability", outcome="mental health", n=n)
        return SimpleNamespace(final_output=AgentDecision(finding_ids=[finding["id"]], gaps=[], clarification=None))

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    async with httpx.AsyncClient(transport=transport()) as client:
        result, calls = await run_agent("literature", "What does research say about walkability?",
                                        ROOT / "data/processed/study", client=client)
    assert isinstance(result, AgentResult) and result.status == "ok" and result.agent == "literature"
    assert isinstance(result.sources[0], PaperSource) and result.findings[0].source_ids == [result.sources[0].id]
    assert [c.name for c in calls] == ["search_pubmed", "record_study"]
    assert AgentResult.model_validate(result.model_dump()).sources[0].doi == result.sources[0].doi
