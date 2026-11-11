"""Orchestrator: plan checks, reconciliation, report rules for comparisons, and the full pipeline with fake
model runs. No network or model calls."""
import json
from types import SimpleNamespace

import httpx
import pytest

from data import ROOT, AgentResult, Finding, PaperSource
from orchestrate import Plan, Task, compare, reconcile, validate_plan
from report import Report, Section, Sentence, check_report, pool, writer_prompt
from verify import CitationCheck, Verdict, VerifierOutput, check_numbers, exclude_bad_sources

SAVED = ROOT / "docs/study/live-agent-checks.json"


@pytest.fixture
def results():
    return [AgentResult.model_validate(item["result"]) for item in json.loads(SAVED.read_text())]


def association(results, adjusted):
    """Pooled (20-metro, with interval) main-effect model from the saved analyst runs."""
    return next(f for r in results for f in r.findings if f.agent == "analyst" and "estimate" in f.evidence
                and f.evidence["adjusted"] is adjusted and not f.evidence.get("moderator") and f.evidence.get("ci95"))


def study(fid="f-lit", outcome="depressive symptoms", exposure="neighborhood walkability",
          effect=None, confidence="literature_moderate", paper_id="doi:10.1/walk"):
    paper = PaperSource(id=paper_id, doi=paper_id.removeprefix("doi:"), title="Walkable neighborhoods and mood",
                        authors=["Ana Lee"], year=2021, venue="Health & Place", url="https://example.org/walk",
                        database="openalex", retrieved_at="2026-11-09")
    evidence = {"paper_id": paper.id, "design": "longitudinal", "exposure": exposure, "outcome": outcome,
                "n": 1200, "effect": effect, "preregistered": None, "quote": "x", "strength": None}
    finding = Finding(id=fid, agent="literature", claim=f"A longitudinal study of {exposure} and {outcome}.",
                      evidence=evidence, source_ids=[paper.id], confidence=confidence, caveats=[])
    return AgentResult(agent="literature", question="q", status="ok", findings=[finding], sources=[paper])


def odds(value, low, high):
    return {"metric": "OR", "value": value, "ci95": [low, high]}


# Plan checks

def test_plan_adds_literature_for_loneliness_and_merges_duplicates():
    plan = Plan(tasks=[Task(role="analyst", sub_question="a", why="w"), Task(role="analyst", sub_question="b", why="w")],
                clarification=None)
    checked, notes = validate_plan("Is walkability associated with loneliness?", plan)
    assert [t.role for t in checked.tasks] == ["analyst", "literature"]
    assert checked.tasks[0].sub_question == "a"
    assert any("duplicate" in n for n in notes) and any("literature" in n for n in notes)


def test_plan_adds_analyst_for_associations_and_caps_tasks():
    plan = Plan(tasks=[Task(role="literature", sub_question="q", why="w")], clarification=None)
    checked, _ = validate_plan("Do walkable tracts have lower distress?", plan)
    assert {t.role for t in checked.tasks} == {"literature", "analyst"}
    every = Plan(tasks=[Task(role=r, sub_question="q", why="w") for r in
                        ["built_environment", "health_demographics", "analyst", "literature"]], clarification=None)
    assert len(validate_plan("Compare everything", every)[0].tasks) == 4


def test_vague_plan_keeps_the_clarification():
    checked, _ = validate_plan("Please analyze the data.", Plan(tasks=[], clarification="Which measure?"))
    assert checked.tasks == [] and checked.clarification == "Which measure?"


# Reconciliation verdicts

def test_verdicts(results):
    adjusted = association(results, True)  # negative, interval excludes zero
    unadjusted = association(results, False)  # interval includes zero
    lit = lambda **kw: study(**kw).findings[0]  # noqa: E731
    assert compare(adjusted, lit(effect=odds(0.8, 0.7, 0.9)))[0] == "agrees"
    assert compare(adjusted, lit(effect=odds(1.3, 1.1, 1.5)))[0] == "conflicts"
    assert compare(adjusted, lit(effect={"metric": "beta", "value": -0.2, "ci95": None}))[0] == "agrees"
    assert compare(adjusted, lit(effect=odds(0.9, 0.7, 1.1)))[0] == "direction_unclear"
    assert compare(adjusted, lit(effect=None))[0] == "direction_unclear"
    assert compare(unadjusted, lit(effect=odds(0.8, 0.7, 0.9)))[0] == "direction_unclear"
    assert compare(adjusted, lit(outcome="loneliness", effect=odds(0.8, 0.7, 0.9)))[0] == "not_comparable"
    assert compare(adjusted, lit(outcome="mental wellbeing", effect=odds(1.2, 1.1, 1.3)))[0] == "not_comparable"
    assert compare(adjusted, lit(exposure="green space", effect=odds(0.8, 0.7, 0.9)))[0] == "not_comparable"
    assert compare(adjusted, lit(exposure="low walkability", effect=odds(1.3, 1.1, 1.5)))[0] == "direction_unclear"


def test_spelled_out_ratio_metrics(results):
    adjusted = association(results, True)
    for metric in ["odds ratio", "Odds Ratio", "aOR", "hazard ratio", "relative risk", "prevalence ratio"]:
        effect = {"metric": metric, "value": 0.9, "ci95": [0.82, 0.996]}
        assert compare(adjusted, study(effect=effect).findings[0])[0] == "agrees", metric
    beta = {"metric": "standardized coefficient", "value": 0.9, "ci95": [0.82, 0.996]}  # null is 0, not 1
    assert compare(adjusted, study(effect=beta).findings[0])[0] == "conflicts"


def test_live_run_crime_study_is_not_comparable(results):
    """From the first live run: the extractor's interpretation in parentheses is not the exposure."""
    lit = study(exposure="perceived neighborhood crime (indicative of greater neighborhood walkability)",
                outcome="elevated symptoms of depression", effect={"metric": "odds ratio", "value": 0.9,
                                                                   "ci95": [0.82, 0.996]}).findings[0]
    verdict, reason = compare(association(results, True), lit)
    assert verdict == "not_comparable" and "perceived neighborhood crime" in reason and "indicative" not in reason
    proxy = study(exposure="street connectivity as a proxy for walkability", effect=odds(0.8, 0.7, 0.9)).findings[0]
    assert compare(association(results, True), proxy)[0] == "not_comparable"


def test_reasons_name_the_actual_cause(results):
    lit = study(effect=odds(0.8, 0.7, 0.9)).findings[0]
    _, pooled = compare(association(results, False), lit)
    assert "interval includes zero" in pooled and "single metro" not in pooled
    single = next(f for r in results for f in r.findings if "estimate" in f.evidence and not f.evidence.get("ci95"))
    _, metro = compare(single, lit)
    assert "single metro" in metro


def test_comparison_claims_read_well_and_pass_number_checks(results):
    lit = study(effect={"metric": "OR", "value": 0.9, "ci95": [0.82, 0.996]})
    pooled = results + [lit, reconcile(results + [lit])]
    findings, _, _ = pool(pooled)
    adjusted = association(results, True)
    comparison = next(f for f in findings.values()
                      if f.agent == "reconciler" and f.evidence["data_finding"] == adjusted.id)
    assert comparison.claim.startswith("The adjusted tract-level estimate (-0.0659, 95% interval")
    assert "(OR 0.9, 95% CI 0.82 to 0.996) point the same way." in comparison.claim
    copied = Report(sections=[Section(heading="Where they agree or disagree", sentences=[
        Sentence(text=comparison.claim, finding_ids=[comparison.id], hedge="exploratory")])], gaps=[])
    assert check_report(copied, findings) == [] and check_numbers(copied, findings) == []


def test_writer_sees_wording_ceiling_and_uncertain_intervals(results):
    from report import writer_item
    unadjusted, adjusted = association(results, False), association(results, True)
    assert writer_item(unadjusted)["interval_includes_zero"] is True
    assert writer_item(adjusted)["interval_includes_zero"] is False
    weak = study(confidence="literature_weak").findings[0]
    assert writer_item(weak)["strongest_wording"] == "exploratory" and "interval_includes_zero" not in writer_item(weak)


def test_long_limitations_are_flagged_past_the_limit(results):
    from report import check_report_items
    from verify import remove_flagged
    findings, _, _ = pool(results)
    data = association(results, True)
    report = Report(sections=[Section(heading="Limitations and caveats", sentences=[
        Sentence(text=f"Caveat {n}.", finding_ids=[data.id], hedge="exploratory") for n in range(9)])], gaps=[])
    problems = check_report(report, findings)
    assert len(problems) == 3 and all("beyond the limit" in p for p in problems)
    trimmed, _ = remove_flagged(report, check_report_items(report, findings))
    assert len(trimmed.sections[0].sentences) == 6


def test_reconcile_findings_carry_sources_and_weakest_confidence(results):
    lit = study(effect=odds(0.8, 0.7, 0.9), confidence="literature_strong")
    out = reconcile(results + [lit])
    assert out.agent == "reconciler" and out.status == "ok"
    assert {f.evidence["data_model"] for f in out.findings} == {"adjusted", "unadjusted"}  # no interactions
    verdicts = {f.evidence["data_finding"]: f.evidence["verdict"] for f in out.findings}
    assert verdicts[association(results, True).id] == "agrees"
    single_metro = [f.id for r in results for f in r.findings if "estimate" in f.evidence and not f.evidence.get("ci95")]
    assert single_metro and all(verdicts[i] == "direction_unclear" for i in single_metro)  # no interval
    adjusted = next(f for f in out.findings if f.evidence["data_finding"] == association(results, True).id)
    assert "doi:10.1/walk" in adjusted.source_ids and adjusted.confidence == "exploratory"
    assert {s.id for s in out.sources} >= set(adjusted.source_ids)


def test_reconcile_without_literature_is_a_gap(results):
    out = reconcile(results)
    assert out.findings == [] and out.status == "insufficient"
    assert out.gaps == ["Literature and data were not compared: no literature finding was available."]


def test_retracted_paper_removes_its_comparisons(results):
    lit = study(effect=odds(0.8, 0.7, 0.9))
    pooled = results + [lit, reconcile(results + [lit])]
    bad = CitationCheck(source_id="doi:10.1/walk", status="retracted", details="Retraction notice")
    kept, _ = exclude_bad_sources(pooled, [bad])
    assert not any(f.agent in {"reconciler", "literature"} for r in kept for f in r.findings)


# Report rules for the agree/disagree section

def test_agree_section_must_cite_comparisons(results):
    lit = study(effect=odds(0.8, 0.7, 0.9))
    pooled = results + [lit, reconcile(results + [lit])]
    findings, _, _ = pool(pooled)
    comparison = next(f for f in findings.values() if f.agent == "reconciler" and f.evidence["data_model"] == "adjusted")
    data = association(results, True)

    def agree(*sentences):
        return Report(sections=[Section(heading="Where they agree or disagree", sentences=list(sentences))], gaps=[])

    good = agree(Sentence(text=f"In this exploratory comparison, the adjusted estimate of "
                               f"{data.evidence['estimate']:.4f} and the study point the same way.",
                          finding_ids=[comparison.id], hedge="exploratory"))
    assert check_report(good, findings) == [] and check_numbers(good, findings) == []
    uncited = agree(Sentence(text="The literature and data agree.", finding_ids=[], hedge="suggests"))
    assert any("without citing" in p for p in check_report(uncited, findings))
    wrong = agree(Sentence(text="The data are negative.", finding_ids=[data.id], hedge="exploratory"))
    assert any("comparison (reconciler)" in p for p in check_report(wrong, findings))
    prompt = writer_prompt("q", pooled)
    assert '"verdict": "agrees"' in prompt and "omit the agree/disagree section" not in prompt
    assert "omit the agree/disagree section" in writer_prompt("q", results)


# Full pipeline with fake model runs

async def test_orchestrate_end_to_end(monkeypatch, results):
    pytest.importorskip("agents")
    import agents
    import app

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    analyst = next(r for r in results if r.agent == "analyst" and any(f.evidence.get("ci95") for f in r.findings))
    lit = study(effect=odds(0.8, 0.7, 0.9))
    data = association(results, True)
    comparison = next(f for f in reconcile([analyst, lit]).findings if f.evidence["data_model"] == "adjusted")

    async def fake_agent(role, question, dataset, **kwargs):
        if role == "built_environment":
            raise TimeoutError("too slow")
        return {"analyst": analyst, "literature": lit}[role], []

    plan = Plan(tasks=[Task(role="analyst", sub_question="Fit both models.", why="association"),
                       Task(role="built_environment", sub_question="Describe walkability.", why="context")],
                clarification=None)
    report = Report(sections=[
        Section(heading="What our data show", sentences=[Sentence(
            text="In this exploratory model the adjusted association is negative.", finding_ids=[data.id],
            hedge="exploratory")]),
        Section(heading="Where they agree or disagree", sentences=[Sentence(
            text="In this exploratory comparison, the adjusted data and one study point the same way.",
            finding_ids=[comparison.id], hedge="exploratory")])], gaps=[])
    prompts = []

    async def fake_run(agent, prompt, **kwargs):
        prompts.append((agent.name, prompt))
        if agent.name == "planner":
            return SimpleNamespace(final_output=plan)
        if agent.name == "writer":
            return SimpleNamespace(final_output=report)
        return SimpleNamespace(final_output=VerifierOutput(verdicts=[
            Verdict(sentence=1, status="supported", reason="ok"), Verdict(sentence=2, status="supported", reason="ok")]))

    monkeypatch.setattr(app, "run_agent", fake_agent)
    monkeypatch.setattr(agents.Runner, "run", fake_run)
    offline = httpx.MockTransport(lambda request: httpx.Response(500, text="offline"))
    events = []
    async with httpx.AsyncClient(transport=offline) as client:
        out = await app.orchestrate("Is walkability associated with loneliness?", ROOT / "data/processed/study",
                                    client=client, hooks=SimpleNamespace(emit=events.append))

    stages = [(e["stage"], e.get("role")) for e in events if e["type"] == "stage"]
    assert stages[0] == ("plan", None) and stages[-2:] == [("compare", None), ("verify", None)]
    assert {r for s, r in stages if s == "agent"} == {"analyst", "built_environment", "literature"}
    by_role = {e["role"]: e for e in events if e.get("stage") == "agent"}
    assert by_role["built_environment"]["status"] == "error" and by_role["analyst"]["findings"] > 0
    assert next(e for e in events if e.get("stage") == "compare")["verdicts"] == {"agrees": 1, "direction_unclear": 1}
    assert [t["role"] for t in out.plan["tasks"]] == ["analyst", "built_environment", "literature"]
    assert {a.role: a.status for a in out.agents} == {"analyst": "ok", "built_environment": "error",
                                                      "literature": "ok"}
    assert out.results[-1].agent == "reconciler"
    assert out.report is not None and out.report.removed == [] and out.report.rounds == 1
    assert "## Where they agree or disagree" in out.report.markdown
    assert "built environment agent did not finish" in out.report.markdown
    compare_section = next(s for s in out.report.view["sections"] if s["heading"] == "Where they agree or disagree")
    assert compare_section["sentences"][0]["verdicts"] == ["agrees"]  # the page shows this as a chip
    assert any("did not finish" in g for g in out.report.view["gaps"])
    writer_prompt_text = next(p for name, p in prompts if name == "writer")
    assert comparison.id in writer_prompt_text


async def test_turn_limit_keeps_findings_already_recorded(monkeypatch):
    """Live run: the literature agent hit MaxTurnsExceeded and the whole agent's work was lost."""
    pytest.importorskip("agents")
    import agents
    from agents import MaxTurnsExceeded
    import app

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")

    async def runs_out(agent, prompt, *, context, **kwargs):
        context.invoke("dataset_metadata")
        context.invoke("profile", columns=["walkability"])
        raise MaxTurnsExceeded("Max turns (8) exceeded")

    monkeypatch.setattr(agents.Runner, "run", runs_out)
    result, calls = await app.run_agent("built_environment", "Describe walkability.", ROOT / "data/processed/study")
    assert result.status == "ok" and result.findings and len(calls) == 2
    assert "limit of 8 model turns" in result.gaps[0]
    assert {s.id for s in result.sources} >= {s for f in result.findings for s in f.source_ids}


async def test_timeout_keeps_findings_already_recorded(monkeypatch):
    pytest.importorskip("agents")
    import asyncio

    import agents
    import app

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")

    async def slow(agent, prompt, *, context, **kwargs):
        context.invoke("dataset_metadata")
        context.invoke("profile", columns=["walkability"])
        await asyncio.sleep(10)  # still "thinking" when the limit hits

    monkeypatch.setattr(agents.Runner, "run", slow)
    result, calls = await app.run_agent("built_environment", "Describe walkability.", ROOT / "data/processed/study",
                                        timeout=0.05)
    assert result.status == "ok" and result.findings and len(calls) == 2
    assert "ran out of time after 0.05 seconds" in result.gaps[0] and app.PARTIAL in result.gaps[0]


async def test_timeout_with_nothing_recorded_is_insufficient_not_an_error(monkeypatch):
    pytest.importorskip("agents")
    import asyncio

    import agents
    import app

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")

    async def stuck(agent, prompt, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(agents.Runner, "run", stuck)
    result, calls = await app.run_agent("literature", "Walkability and depression studies?", ROOT, timeout=0.05)
    assert result.status == "insufficient" and result.findings == [] and calls == []
    assert "0 finding(s)" in result.gaps[0]


async def test_study_uses_partial_literature_and_marks_it(monkeypatch, results):
    """A literature agent cut off after recording one study still contributes it to the report."""
    pytest.importorskip("agents")
    import agents
    import app

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    analyst = next(r for r in results if r.agent == "analyst" and any(f.evidence.get("ci95") for f in r.findings))
    lit = study(effect=odds(0.8, 0.7, 0.9))
    cut = lit.model_copy(update={"gaps": [f"The literature agent ran out of time after 360 seconds; its 1 finding(s) "
                                          f"recorded so far are kept, and {app.PARTIAL}."]})

    async def fake_agent(role, question, dataset, **kwargs):
        assert kwargs["timeout"] == app.agent_timeout(role)  # the limit is passed in, not applied outside
        return {"analyst": analyst, "literature": cut}[role], []

    plan = Plan(tasks=[Task(role="analyst", sub_question="Fit both models.", why="a"),
                       Task(role="literature", sub_question="Studies of walkability and depression.", why="b")],
                clarification=None)
    data = association(results, True)
    report = Report(sections=[Section(heading="What our data show", sentences=[Sentence(
        text="In this exploratory model the adjusted association is negative.", finding_ids=[data.id],
        hedge="exploratory")])], gaps=[])

    async def fake_run(agent, prompt, **kwargs):
        if agent.name == "planner":
            return SimpleNamespace(final_output=plan)
        if agent.name == "writer":
            return SimpleNamespace(final_output=report)
        return SimpleNamespace(final_output=VerifierOutput(verdicts=[Verdict(sentence=1, status="supported", reason="ok")]))

    monkeypatch.setattr(app, "run_agent", fake_agent)
    monkeypatch.setattr(agents.Runner, "run", fake_run)
    offline = httpx.MockTransport(lambda request: httpx.Response(500, text="offline"))
    async with httpx.AsyncClient(transport=offline) as client:
        out = await app.orchestrate("Walkability and depression?", ROOT / "data/processed/study", client=client)
    status = {a.role: a for a in out.agents}
    assert status["literature"].status == "partial" and "ran out of time" in status["literature"].error
    assert status["analyst"].status == "ok"
    assert any(f.agent == "reconciler" for r in out.results for f in r.findings)  # partial study still compared
    assert "ran out of time" in out.report.markdown  # listed under Open gaps


async def test_orchestrate_asks_when_the_plan_is_empty(monkeypatch):
    pytest.importorskip("agents")
    import agents
    import app

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")

    async def fake_run(agent, prompt, **kwargs):
        assert agent.name == "planner"
        return SimpleNamespace(final_output=Plan(tasks=[], clarification="Which measure should I study?"))

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    out = await app.orchestrate("Please analyze the data.", ROOT / "data/processed/study")
    assert out.report is None and out.clarification == "Which measure should I study?" and out.agents == []



async def test_explicit_baseline_request_is_not_satisfied_by_interactions(monkeypatch):
    import agents
    import app
    from data import AgentDecision

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")

    async def run(agent, prompt, *, context, **kwargs):
        context.invoke("dataset_metadata")
        interaction = context.invoke("association", adjusted=True, moderator="poverty_pct")
        return SimpleNamespace(final_output=AgentDecision(finding_ids=[interaction["id"]],
                                                          gaps=[], clarification=None))

    monkeypatch.setattr(agents.Runner, "run", run)
    result, _ = await app.run_agent(
        "analyst", "Compare unadjusted and adjusted walkability–distress associations and the poverty interaction.",
        ROOT / "data/processed/study", metro="12060")
    models = [f.evidence for f in result.findings]
    assert {m["adjusted"] for m in models if not m["moderator"]} == {False, True}
    assert any(m["moderator"] == "poverty_pct" for m in models)
    assert all(m["metro_filter"] == "12060" for m in models)



async def test_default_chat_collects_parallel_research_and_one_followup(monkeypatch):
    import asyncio

    import agents
    import app
    import verify
    from data import ToolCall

    monkeypatch.setenv("SIDEWALK_MODEL", "vertex_ai/gemini-3.5-flash-lite")
    initial = Plan(tasks=[
        Task(role="analyst", sub_question="Fit the distress model.", why="data"),
        Task(role="literature", sub_question="Find walkability studies.", why="literature"),
    ], clarification=None)
    targeted = Task(role="literature", sub_question="Search for a depression cohort.", why="Fill the evidence gap")
    revised = Plan(tasks=[targeted, targeted], clarification=None)
    plans = iter([initial, revised])
    started = set()
    both_started = asyncio.Event()

    async def planner(agent, prompt, **kwargs):
        plan = next(plans)
        if plan is revised:
            assert "No directly relevant paper found" in prompt
        return SimpleNamespace(final_output=plan)

    async def research(role, question, dataset, **kwargs):
        if question != targeted.sub_question:
            started.add(role)
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 1)
        result = study().model_copy(update={"question": question})
        if role == "analyst":
            result = AgentResult(agent=role, question=question, status="ok", findings=[], sources=[])
        elif question != targeted.sub_question:
            result = result.model_copy(update={"findings": [], "sources": [], "status": "insufficient",
                                               "gaps": ["No directly relevant paper found."]})
        call = ToolCall(agent=role, name="search_pubmed" if role == "literature" else "dataset_metadata",
                        arguments={}, result={}, duration_ms=1)
        return result, [call]

    async def citations(*args, **kwargs):
        return []

    async def writer(question, results, **kwargs):
        paper_results = next(r for r in results if r.agent == "literature")
        assert len(paper_results.findings) == 1 and not paper_results.gaps
        return app.VerifiedReport(markdown="Report with a verified paper.", report={}, citations=[], removed=[], rounds=1)

    monkeypatch.setattr(agents.Runner, "run", planner)
    monkeypatch.setattr(app, "run_agent", research)
    monkeypatch.setattr(verify, "check_citations", citations)
    monkeypatch.setattr(app, "write_verified_report", writer)
    monkeypatch.setattr(app, "save_run", lambda *args: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "How does walkability relate to distress and depression research?"})
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Report with a verified paper."
    assert [a["round"] for a in body["result"]["agents"]] == [1, 1, 2]
    assert body["result"]["followup_tasks"] == [targeted.model_dump()]
    assert [c["agent"] for c in body["tool_calls"]] == ["analyst", "literature", "literature"]
