"""Writer rules and rendering on real saved agent results. No model calls."""
import json
from types import SimpleNamespace

import pytest

from data import ROOT, AgentResult, Finding, PaperSource
from report import Report, Section, Sentence, check_report, pool, render, writer_prompt

SAVED = ROOT / "docs/study/live-agent-checks.json"


@pytest.fixture
def results():
    return [AgentResult.model_validate(item["result"]) for item in json.loads(SAVED.read_text())]


@pytest.fixture
def analyst(results):
    """The adjusted/unadjusted model finding from a saved analyst run (exploratory)."""
    return next(f for r in results for f in r.findings if f.agent == "analyst" and f.confidence == "exploratory")


def report(heading, *sentences, gaps=()):
    return Report(sections=[Section(heading=heading, sentences=list(sentences))], gaps=list(gaps))


def literature_result():
    paper = PaperSource(id="doi:10.1/walk", doi="10.1/walk", title="Walkable neighborhoods and depression",
                        authors=["Ana Lee", "Bo Kim", "Cy Park", "Di Roe"], year=2021, venue="Health & Place",
                        url="https://example.org/walk", database="openalex", preprint=True,
                        retrieved_at="2026-11-09")
    finding = Finding(id="f-lit", agent="literature", claim="A cohort study reported lower depression odds.",
                      evidence={"paper_id": paper.id}, source_ids=[paper.id],
                      confidence="literature_moderate", caveats=[])
    return AgentResult(agent="literature", question="q", status="ok", findings=[finding], sources=[paper])


def test_valid_report_has_no_problems(results, analyst):
    findings, _, _ = pool(results)
    draft = report("What our data show",
                   Sentence(text="In this exploratory model the adjusted association is negative.",
                            finding_ids=[analyst.id], hedge="exploratory"))
    assert check_report(draft, findings) == []


def test_unknown_finding_is_flagged(results):
    findings, _, _ = pool(results)
    draft = report("What our data show", Sentence(text="Made up.", finding_ids=["f-invented"], hedge="unknown"))
    assert "not given" in check_report(draft, findings)[0]


def test_wording_stronger_than_evidence_is_flagged(results, analyst):
    findings, _, _ = pool(results)
    draft = report("What our data show",
                   Sentence(text="Walkable tracts have less distress.", finding_ids=[analyst.id], hedge="established"))
    assert any("only supports 'exploratory'" in p for p in check_report(draft, findings))


def test_causal_wording_about_data_is_flagged(results, analyst):
    findings, _, _ = pool(results)
    draft = report("What our data show",
                   Sentence(text="Walkability reduces distress.", finding_ids=[analyst.id], hedge="exploratory"))
    assert any("causal" in p for p in check_report(draft, findings))


def test_uncited_claim_and_wrong_section_are_flagged(results, analyst):
    findings, _, _ = pool(results)
    draft = Report(sections=[
        Section(heading="What the literature says",
                sentences=[Sentence(text="Studies agree.", finding_ids=[analyst.id], hedge="exploratory")]),
        Section(heading="Limitations and caveats",
                sentences=[Sentence(text="Data are modeled.", finding_ids=[], hedge="established")]),
    ], gaps=[])
    problems = check_report(draft, findings)
    assert any("literature section cites a data finding" in p for p in problems)
    assert any("without citing" in p for p in problems)


def test_render_citations_gaps_and_references(results, analyst):
    draft = report("What our data show",
                   Sentence(text="In this exploratory model the association is negative.",
                            finding_ids=[analyst.id], hedge="exploratory"),
                   Sentence(text="The same model also excludes incomplete tracts.",
                            finding_ids=[analyst.id], hedge="exploratory"),
                   gaps=["No literature findings were provided."])
    markdown = render(draft, results)
    n = len(analyst.source_ids)
    citation = "[" + ", ".join(str(i) for i in range(1, n + 1)) + "]"
    assert markdown.count(citation + ".") == 2  # same sources keep the same numbers, before the period
    references = markdown.split("## References")[1].strip().splitlines()
    assert len(references) == n  # each cited source listed once
    _, _, gaps = pool(results)
    assert gaps and all(f"- {g}" in markdown for g in gaps)
    assert "- No literature findings were provided." in markdown


def test_paper_source_rendering_and_literature_section():
    lit = literature_result()
    draft = report("What the literature says",
                   Sentence(text="One cohort study suggests lower depression odds.", finding_ids=["f-lit"],
                            hedge="suggests"))
    assert check_report(draft, pool([lit])[0]) == []
    markdown = render(draft, [lit])
    assert "Ana Lee, Bo Kim, Cy Park et al. (2021)" in markdown
    assert "[Preprint] https://doi.org/10.1/walk" in markdown


def test_prompt_lists_findings_once_and_flags_missing_literature(results):
    prompt = writer_prompt("Is walkability associated with distress?", results, feedback=["fix this"])
    findings, _, _ = pool(results)
    assert all(prompt.count(f'"id": "{f}"') == 1 for f in findings)
    assert "No literature findings were provided." in prompt
    assert "- fix this" in prompt
    assert "Raw file SHA256s" not in prompt and len(prompt) < 30_000


async def test_write_report_revises_then_rejects_unknown_ids(monkeypatch, results, analyst):
    pytest.importorskip("agents")
    import agents
    from app import write_report

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    bad = report("What our data show", Sentence(text="x.", finding_ids=["f-invented"], hedge="unknown"))
    good = report("What our data show", Sentence(text="In this exploratory model it is negative.",
                                                 finding_ids=[analyst.id], hedge="exploratory"))
    prompts = []

    def fake_run(drafts):
        async def run(agent, prompt, **kwargs):
            prompts.append(prompt)
            return SimpleNamespace(final_output=drafts.pop(0))
        return run

    monkeypatch.setattr(agents.Runner, "run", fake_run([bad, good]))
    _, markdown, problems = await write_report("q", results)
    assert problems == [] and "not given" in prompts[1]  # second draft saw the first draft's problems
    assert "## References" in markdown

    monkeypatch.setattr(agents.Runner, "run", fake_run([bad, bad]))
    with pytest.raises(ValueError, match="not given"):
        await write_report("q", results)
