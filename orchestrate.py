"""Orchestrator rules: plan checks and literature–data reconciliation. No model calls; app.py runs the planner.

The planner (an LLM) proposes which agents to run; Python makes sure the roles a question needs are present.
Reconciliation pairs data associations with literature findings and states, in Python, whether they agree,
conflict, or cannot be compared. The Writer must cite these comparison findings in "Where they agree or
disagree", and the Verifier checks them like any other finding.
"""
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from data import AgentResult, Finding, Role
from report import CEILING, HEDGE_RANK


# Question breakdown and delegation

class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    sub_question: str  # self-contained part of the question for this agent
    why: str  # short reason, shown in logs and the page


class Plan(BaseModel):
    """The planner's output_type."""

    tasks: list[Task]
    clarification: str | None


NEEDS_LITERATURE = re.compile(r"\b(lonel\w*|social(?:ly)?\s+(?:connect\w*|isolat\w*|support)|why|mechanism\w*|"
                              r"explain\w*|research|studies|study|literature|published)\b", re.IGNORECASE)
NEEDS_ANALYST = re.compile(r"\b(associat\w*|relat\w*|link\w*|correlat\w*|compar\w*|differ\w*|adjust\w*|"
                           r"predict\w*|lower|higher)\b", re.IGNORECASE)
MAX_TASKS = 4


def validate_plan(question: str, plan: Plan) -> tuple[Plan, list[str]]:
    """One task per role, required roles added, at most MAX_TASKS. Returns (plan, notes for the log)."""
    notes: list[str] = []
    by_role: dict[str, Task] = {}
    for task in plan.tasks:
        if task.role in by_role:
            notes.append(f"Merged a duplicate {task.role} task.")
        else:
            by_role[task.role] = task
    if not by_role and plan.clarification:
        return plan.model_copy(update={"tasks": []}), notes
    if NEEDS_LITERATURE.search(question) and "literature" not in by_role:
        by_role["literature"] = Task(role="literature", sub_question=question,
                                     why="Added: the data cannot answer loneliness, social connection or mechanisms.")
        notes.append("Added a literature task.")
    if NEEDS_ANALYST.search(question) and "analyst" not in by_role:
        by_role["analyst"] = Task(role="analyst", sub_question=question,
                                  why="Added: the question asks about an association.")
        notes.append("Added an analyst task.")
    tasks = list(by_role.values())
    if len(tasks) > MAX_TASKS:
        notes.append(f"Dropped {len(tasks) - MAX_TASKS} task(s) over the limit of {MAX_TASKS}.")
    return plan.model_copy(update={"tasks": tasks[:MAX_TASKS], "clarification": None if tasks else plan.clarification}), notes


# Reconciling literature and data findings

Verdict = Literal["agrees", "conflicts", "not_comparable", "direction_unclear"]

WALKABILITY = re.compile(r"walkab|walk score|walkscore", re.IGNORECASE)
REVERSED = re.compile(r"\b(low|lower|poor|poorer|less|lack of)\b", re.IGNORECASE)  # "low walkability"
# Outcomes where higher means worse, like modeled frequent mental distress. Wellbeing, quality of life or general
# "mental health" scores run the other way or are ambiguous, so they are treated as not comparable.
ADVERSE = re.compile(r"distress|depress|anxi|psychological symptoms|mental disorder|poor mental health", re.IGNORECASE)
SOCIAL = re.compile(r"lonel|social|isolat|cohesion", re.IGNORECASE)
RATIO_METRICS = {"OR", "AOR", "RR", "HR", "IRR", "PR"}


def _data_direction(evidence: dict) -> int:
    """-1: higher walkability with less distress; +1: more; 0: interval missing or includes zero."""
    ci = evidence.get("ci95")
    if not ci or ci[0] <= 0 <= ci[1]:
        return 0
    return -1 if evidence["estimate"] < 0 else 1


def _literature_direction(effect: dict | None) -> int | None:
    """Same scale as _data_direction; None when the study reports no effect."""
    if not effect:
        return None
    null = 1.0 if effect["metric"].strip().upper() in RATIO_METRICS else 0.0
    ci = effect.get("ci95")
    if ci and ci[0] <= null <= ci[1]:
        return 0
    if effect["value"] == null:
        return 0
    return -1 if effect["value"] < null else 1


def compare(data: Finding, literature: Finding) -> tuple[Verdict, str]:
    """Verdict and reason for one data association and one literature finding."""
    study = literature.evidence
    exposure, outcome = study.get("exposure", ""), study.get("outcome", "")
    if not WALKABILITY.search(exposure):
        return "not_comparable", f"the study's exposure is {exposure or 'not stated'}, not walkability"
    if SOCIAL.search(outcome):
        return "not_comparable", (f"the study measured {outcome}; our data measure modeled frequent mental distress, "
                                  "not loneliness or social connection")
    if not ADVERSE.search(outcome):
        return "not_comparable", f"the study measured {outcome or 'an unstated outcome'}, not mental distress"
    if REVERSED.search(exposure):
        return "direction_unclear", f"the study states its exposure in reverse ({exposure})"
    ours, theirs = _data_direction(data.evidence), _literature_direction(study.get("effect"))
    if theirs is None:
        return "direction_unclear", "the study's abstract reports no effect size to compare"
    if ours == 0:
        return "direction_unclear", ("our interval includes zero or is not reported (single metro), "
                                     "so the data direction is uncertain")
    if theirs == 0:
        return "direction_unclear", "the study's interval includes no association"
    if ours == theirs:
        return "agrees", "both find the same direction of association"
    return "conflicts", "the data and the study find opposite directions of association"


def _weakest(*confidences: str) -> str:
    return min(confidences, key=lambda c: HEDGE_RANK[CEILING[c]])


def reconcile(results: list[AgentResult]) -> AgentResult:
    """Comparison findings for every main-effect data association × literature finding pair."""
    data = [f for r in results for f in r.findings
            if f.agent == "analyst" and "estimate" in f.evidence and not f.evidence.get("moderator")]
    literature = [f for r in results for f in r.findings if f.agent == "literature"]
    sources = {s.id: s for r in results for s in r.sources}
    findings: list[Finding] = []
    gaps: list[str] = []
    if not data or not literature:
        missing = "data association" if not data else "literature finding"
        gaps.append(f"Literature and data were not compared: no {missing} was available.")
    for d in data:
        model = "adjusted" if d.evidence.get("adjusted") else "unadjusted"
        for lit in literature:
            verdict, reason = compare(d, lit)
            study = lit.evidence
            design = str(study.get("design", "other")).replace("_", " ")
            body = {
                "agent": "reconciler",
                "claim": (f"Comparison of the {model} tract-level walkability–distress association with a {design} "
                          f"study of {study.get('outcome', 'an unstated outcome')}: {verdict.replace('_', ' ')} "
                          f"({reason})."),
                "evidence": {"verdict": verdict, "reason": reason, "data_finding": d.id, "literature_finding": lit.id,
                             "data_model": model, "data_estimate": d.evidence["estimate"],
                             "data_ci95": d.evidence.get("ci95"), "literature_design": study.get("design"),
                             "literature_outcome": study.get("outcome"), "literature_effect": study.get("effect")},
                "source_ids": list(dict.fromkeys(d.source_ids + lit.source_ids)),
                "confidence": _weakest(d.confidence, lit.confidence),
                "caveats": ["Compares a tract-level association with a study of different people, places and design; "
                            "agreement in direction does not establish causation or individual-level effects."],
            }
            identifier = "cmp-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
            findings.append(Finding(id=identifier, **body))
    cited = dict.fromkeys(s for f in findings for s in f.source_ids)
    return AgentResult(agent="reconciler", question="Compare literature and data findings",
                       status="ok" if findings else "insufficient", findings=findings,
                       sources=[sources[s] for s in cited if s in sources], gaps=gaps)
