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
# What a study measured, without the extractor's own interpretation: "perceived crime (indicative of walkability)"
# is a crime measure, not a walkability measure.
PROXY = re.compile(r"\(.*?\)|\b(?:indicative of|as a proxy for|a proxy for|proxy for|reflecting|used to indicate)\b.*",
                   re.IGNORECASE)
REVERSED = re.compile(r"\b(low|lower|poor|poorer|less|lack of)\b", re.IGNORECASE)  # "low walkability"
# Outcomes where higher means worse, like modeled frequent mental distress. Wellbeing, quality of life or general
# "mental health" scores run the other way or are ambiguous, so they are treated as not comparable.
ADVERSE = re.compile(r"distress|depress|anxi|psychological symptoms|mental disorder|poor mental health", re.IGNORECASE)
SOCIAL = re.compile(r"lonel|social|isolat|cohesion", re.IGNORECASE)
# Ratio metrics have no association at 1; differences, coefficients and correlations at 0.
RATIO = re.compile(r"\b(?:a?or|odds[- ]ratios?|rr|relative risks?|risk ratios?|a?hr|hazard ratios?|irr|"
                   r"incidence rate ratios?|pr|prevalence ratios?|rate ratios?)\b", re.IGNORECASE)


def null_value(metric: str) -> float:
    return 1.0 if RATIO.search(metric) else 0.0


def main_exposure(exposure: str) -> str:
    return re.sub(r"\s+", " ", PROXY.sub("", exposure)).strip(" ,;")


def _literature_direction(effect: dict | None) -> int | None:
    """-1: higher walkability with less of the outcome; +1: more; 0: no association; None: no effect reported."""
    if not effect:
        return None
    null = null_value(effect["metric"])
    ci = effect.get("ci95")
    if ci and ci[0] <= null <= ci[1]:
        return 0
    if effect["value"] == null:
        return 0
    return -1 if effect["value"] < null else 1


def compare(data: Finding, literature: Finding) -> tuple[Verdict, str]:
    """Verdict and reason for one data association and one literature finding."""
    study = literature.evidence
    exposure, outcome = main_exposure(study.get("exposure", "")), study.get("outcome", "")
    if not WALKABILITY.search(exposure):
        return "not_comparable", f"the study measured {exposure or 'an unstated exposure'}, not walkability"
    if SOCIAL.search(outcome):
        return "not_comparable", (f"the study measured {outcome}, while our data measure modeled frequent mental "
                                  "distress, not loneliness or social connection")
    if not ADVERSE.search(outcome):
        return "not_comparable", f"the study measured {outcome or 'an unstated outcome'}, not mental distress"
    if REVERSED.search(exposure):
        return "direction_unclear", f"the study states its exposure in reverse ({exposure})"
    ci = data.evidence.get("ci95")
    if ci is None:
        return "direction_unclear", "the data have no confidence interval (single metro), so their direction is uncertain"
    if ci[0] <= 0 <= ci[1]:
        return "direction_unclear", "the data interval includes zero, so their direction is uncertain"
    theirs = _literature_direction(study.get("effect"))
    if theirs is None:
        return "direction_unclear", "no single effect estimate was recorded for comparison"
    if theirs == 0:
        return "direction_unclear", "the study's interval includes no association"
    ours = -1 if data.evidence["estimate"] < 0 else 1
    if ours == theirs:
        return "agrees", "both point the same way"
    return "conflicts", "they point in opposite directions"


def _number(value: float) -> str:
    return f"{value:+.4f}".replace("+", "") if abs(value) < 1 else f"{value:g}"


def _data_text(model: str, evidence: dict) -> str:
    text = f"the {model} tract-level estimate ({_number(evidence['estimate'])}"
    ci = evidence.get("ci95")
    if ci:
        text += f", 95% interval {_number(ci[0])} to {_number(ci[1])}"
    return text + ")"


def _study_text(study: dict) -> str:
    design = str(study.get("design", "other")).replace("_", " ")
    text = f"a {design} study of {study.get('outcome') or 'an unstated outcome'}"
    effect = study.get("effect")
    if effect:
        text += f" ({effect['metric']} {effect['value']:g}"
        if effect.get("ci95"):
            text += f", 95% CI {effect['ci95'][0]:g} to {effect['ci95'][1]:g}"
        text += ")"
    return text


def comparison_claim(model: str, data: dict, study: dict, verdict: Verdict, reason: str) -> str:
    """A sentence a reader can use, with both sides' numbers; verdict wording is fixed per verdict."""
    ours, theirs = _data_text(model, data), _study_text(study)
    first = ours[0].upper() + ours[1:]
    if verdict == "agrees":
        return f"{first} and {theirs} point the same way."
    if verdict == "conflicts":
        return f"{first} and {theirs} point in opposite directions."
    if verdict == "not_comparable":
        return f"{first} cannot be compared with {theirs}: {reason}."
    return f"Whether {ours} and {theirs} agree is unclear: {reason}."


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
            body = {
                "agent": "reconciler",
                "claim": comparison_claim(model, d.evidence, study, verdict, reason),
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
