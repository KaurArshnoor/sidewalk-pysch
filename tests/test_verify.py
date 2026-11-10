"""Verifier checks on recorded Crossref/PubMed responses and saved agent results. No network or model calls.

Refresh the recorded responses with: uv run python -m scripts.record_verify_responses
"""
import json
from types import SimpleNamespace

import httpx
import pytest

import verify
from data import ROOT, AgentResult, Finding, PaperSource
from report import Report, Section, Sentence, pool
from scripts.record_verify_responses import SOURCES
from verify import (Verdict, VerifierOutput, check_citations, check_numbers, exclude_bad_sources,
                    verdict_items, verifier_prompt)

RECORDED = json.loads((ROOT / "tests/verify_responses.json").read_text())
SAVED = ROOT / "docs/study/live-agent-checks.json"
BY_ID = {s.id: s for s in SOURCES}


def transport(calls=None, overrides=None):
    def handle(request: httpx.Request):
        url = str(request.url)
        if calls is not None:
            calls.append(url)
        if overrides and url in overrides:
            return overrides[url](request)
        response = RECORDED.get(url)
        if response is None:
            return httpx.Response(500, text="not recorded")
        if response["json"] is None:
            return httpx.Response(response["status"], text=response["text"] or "")
        return httpx.Response(response["status"], json=response["json"])
    return httpx.MockTransport(handle)


async def checks(sources, **kwargs):
    async with httpx.AsyncClient(transport=transport(**kwargs)) as client:
        return {c.source_id: c for c in await check_citations(sources, client)}


@pytest.fixture
def results():
    return [AgentResult.model_validate(item["result"]) for item in json.loads(SAVED.read_text())]


@pytest.fixture
def analyst(results):
    return next(f for r in results for f in r.findings if f.agent == "analyst" and "estimate" in f.evidence
                and f.evidence["adjusted"] and not f.evidence.get("moderator"))


def lit(source: PaperSource, fid="f-lit"):
    finding = Finding(id=fid, agent="literature", claim="A study reported an association.",
                      evidence={"paper_id": source.id}, source_ids=[source.id],
                      confidence="literature_moderate", caveats=[])
    return AgentResult(agent="literature", question="q", status="ok", findings=[finding], sources=[source])


# Citations

async def test_recorded_valid_retracted_and_missing():
    found = await checks(SOURCES)
    assert found["valid"].status == "ok"
    assert found["retracted"].status == "retracted"
    assert found["missing"].status == "not_found"


async def test_crossref_alone_detects_retraction():
    doi_only = BY_ID["retracted"].model_copy(update={"pmid": None})
    found = await checks([doi_only])
    assert found["retracted"].status == "retracted" and "Crossref" in found["retracted"].details


async def test_title_and_year_mismatch():
    wrong = BY_ID["valid"].model_copy(update={"title": "Green space and anxiety in adolescents", "year": 2010})
    check = (await checks([wrong]))["valid"]
    assert check.status == "metadata_mismatch"
    assert "title" in check.details and "year" in check.details


async def test_dataset_sources_are_not_looked_up(results):
    calls = []
    _, sources, _ = pool(results)
    found = await checks(list(sources.values()), calls=calls)
    assert calls == [] and {c.status for c in found.values()} == {"unchecked"}


async def test_network_failure_keeps_the_study():
    def timeout(request):
        raise httpx.ConnectTimeout("timed out", request=request)
    overrides = {url: timeout for url in RECORDED}
    assert (await checks([BY_ID["valid"]], overrides=overrides))["valid"].status == "lookup_failed"
    crossref_down = {url: timeout for url in RECORDED if "crossref" in url}
    check = (await checks([BY_ID["retracted"]], overrides=crossref_down))["retracted"]
    assert check.status == "retracted" and "DOI lookup failed" in check.details  # PubMed still decides


async def test_rate_limit_is_retried(monkeypatch):
    async def no_wait(seconds):
        pass
    monkeypatch.setattr(verify.asyncio, "sleep", no_wait)
    url = next(u for u in RECORDED if "ypmed" in u)
    attempts = []

    def limited(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=RECORDED[url]["json"])
    assert (await checks([BY_ID["valid"]], overrides={url: limited}))["valid"].status == "ok"
    assert len(attempts) == 2


async def test_excluded_findings_and_gaps(results):
    found = await checks([BY_ID["retracted"], BY_ID["valid"]])
    pooled = results + [lit(BY_ID["retracted"], "f-bad"), lit(BY_ID["valid"], "f-good")]
    kept, gaps = exclude_bad_sources(pooled, list(found.values()))
    ids = {f.id for r in kept for f in r.findings}
    assert "f-bad" not in ids and "f-good" in ids
    assert all(s.id != "retracted" for r in kept for s in r.sources)
    assert len(gaps) == 1 and gaps[0].startswith("Excluded retracted study")


# Numbers and verdicts

def one(text, finding_ids, hedge="exploratory"):
    return Report(sections=[Section(heading="What our data show",
                                    sentences=[Sentence(text=text, finding_ids=finding_ids, hedge=hedge)])], gaps=[])


def test_numbers_must_come_from_cited_findings(results, analyst):
    findings, _, _ = pool(results)
    estimate, n = analyst.evidence["estimate"], analyst.evidence["n_complete"]
    good = one(f"In this exploratory model the estimate is −{abs(estimate):.4f} across {n:,} tracts in 20 metros.",
               [analyst.id])
    assert check_numbers(good, findings) == []
    invented = one("In this exploratory model the estimate is -0.12 across 31,500 tracts.", [analyst.id])
    problems = check_numbers(invented, findings)
    assert problems and "-0.12" in problems[0][2] and "31,500" in problems[0][2]


def test_missing_and_negative_verdicts_are_problems(results, analyst):
    findings, _, _ = pool(results)
    report = Report(sections=[Section(heading="What our data show", sentences=[
        Sentence(text="A.", finding_ids=[analyst.id], hedge="exploratory"),
        Sentence(text="B.", finding_ids=[analyst.id], hedge="exploratory"),
        Sentence(text="C.", finding_ids=[analyst.id], hedge="exploratory")])], gaps=[])
    output = VerifierOutput(verdicts=[Verdict(sentence=1, status="supported", reason="ok"),
                                      Verdict(sentence=2, status="overstated", reason="too strong")])
    items = verdict_items(report, output)
    assert [(i, m.split(":")[0]) for _, i, m in items] == [(1, "overstated"), (2, "the verifier returned no verdict")]
    prompt = verifier_prompt(report, findings)
    assert '"sentence": 3' in prompt and "Raw file SHA256s" not in prompt and '"scope"' not in prompt


# Full write → verify loop with fake model runs

def fake_runner(writer_drafts, verifier_outputs, prompts):
    async def run(agent, prompt, **kwargs):
        prompts.append((agent.name, prompt))
        queue = writer_drafts if agent.name == "writer" else verifier_outputs
        return SimpleNamespace(final_output=queue.pop(0) if len(queue) > 1 else queue[0])
    return run


async def test_revision_fixes_the_report(monkeypatch, results, analyst):
    pytest.importorskip("agents")
    import agents
    from app import write_verified_report

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    draft = one("In this exploratory model the association is negative.", [analyst.id])
    bad, ok = (VerifierOutput(verdicts=[Verdict(sentence=1, status=s, reason=r)])
               for s, r in [("overstated", "ignores the interval"), ("supported", "matches")])
    prompts = []
    monkeypatch.setattr(agents.Runner, "run", fake_runner([draft], [bad, ok], prompts))
    async with httpx.AsyncClient(transport=transport()) as client:
        pooled = results + [lit(BY_ID["retracted"], "f-bad")]
        out = await write_verified_report("q", pooled, client=client)
    assert out.rounds == 2 and out.removed == []
    writer_prompts = [p for name, p in prompts if name == "writer"]
    assert "ignores the interval" in writer_prompts[1]  # the Writer saw the Verifier's feedback
    assert all("f-bad" not in p for p in writer_prompts)  # retracted study never reached the Writer
    assert "Excluded retracted study" in out.markdown and "## Verification" in out.markdown


async def test_still_flagged_sentences_are_removed(monkeypatch, results, analyst):
    pytest.importorskip("agents")
    import agents
    from app import write_verified_report

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    report = Report(sections=[Section(heading="What our data show", sentences=[
        Sentence(text="In this exploratory model the association is negative.", finding_ids=[analyst.id],
                 hedge="exploratory"),
        Sentence(text="Walkable places clearly protect mental health.", finding_ids=[analyst.id],
                 hedge="exploratory")])], gaps=[])
    verdicts = VerifierOutput(verdicts=[Verdict(sentence=1, status="supported", reason="matches"),
                                        Verdict(sentence=2, status="overstated", reason="causal and certain")])
    monkeypatch.setattr(agents.Runner, "run", fake_runner([report], [verdicts], []))
    async with httpx.AsyncClient(transport=transport()) as client:
        out = await write_verified_report("q", results, client=client)
    assert out.rounds == 2 and len(out.removed) == 1
    assert "clearly protect" not in out.markdown.split("## Open gaps")[0]
    assert "Removed an unverified statement" in out.markdown
    assert "association is negative" in out.markdown
