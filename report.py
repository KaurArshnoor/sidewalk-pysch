"""Report format, checks and rendering for the Writer. No model calls; app.py runs the Writer agent.

The Writer chooses and words findings; Python enforces the rules: only given findings, wording no
stronger than the evidence, no causal language about tract associations, citations and gaps.
"""
import json
import re
from typing import Literal

from pydantic import BaseModel

from data import AgentResult, Finding, PaperSource, Source

Hedge = Literal["established", "suggests", "exploratory", "unknown"]
Heading = Literal["Question", "What the literature says", "What our data show",
                  "Where they agree or disagree", "Limitations and caveats"]


class Sentence(BaseModel):
    text: str  # no citation brackets; render() adds them
    finding_ids: list[str]
    hedge: Hedge


class Section(BaseModel):
    heading: Heading
    sentences: list[Sentence]


class Report(BaseModel):
    """The Writer's output_type. Open gaps and References are added by render()."""

    sections: list[Section]
    gaps: list[str]


HEDGE_RANK = {"unknown": 0, "exploratory": 1, "suggests": 2, "established": 3}
CEILING: dict[str, Hedge] = {  # strongest wording each finding confidence allows
    "descriptive": "established",
    "exploratory": "exploratory",
    "insufficient": "unknown",
    "literature_strong": "established",
    "literature_moderate": "suggests",
    "literature_weak": "exploratory",
}
CAUSAL = re.compile(r"\b(causes?|caused|causal(?:ly)?|leads? to|led to|improves?|improved|reduces?|reduced|"
                    r"effects? of|because of|results? in|resulted in|drives?)\b", re.IGNORECASE)
UNCITED_OK = {"Question"}  # framing only; comparisons must cite reconciler findings
MAX_LIMITATIONS = 6  # the prompt asks for at most 5; one sentence of slack before it counts as a problem
UNKNOWN = "cites findings you were not given"
PROVENANCE = "Raw file SHA256s:"  # checksum source notes: kept on findings, not useful to the Writer


def pool(results: list[AgentResult]) -> tuple[dict[str, Finding], dict[str, Source | PaperSource], list[str]]:
    """Findings, sources and gaps across agents, de-duplicated in order."""
    findings = {f.id: f for r in results for f in r.findings}
    sources = {s.id: s for r in results for s in r.sources}
    gaps = list(dict.fromkeys(g for r in results for g in r.gaps))
    return findings, sources, gaps


def where(report: Report, section: int, sentence: int) -> str:
    return f"{report.sections[section].heading} sentence {sentence + 1}"


def check_report_items(report: Report, findings: dict[str, Finding]) -> list[tuple[int, int, str]]:
    """Rule violations as (section index, sentence index, message)."""
    problems = []
    for si, section in enumerate(report.sections):
        if section.heading == "Limitations and caveats" and len(section.sentences) > MAX_LIMITATIONS:
            # Flag every extra sentence, so a draft that stays too long is trimmed when flagged sentences are removed.
            problems += [(si, i, f"beyond the limit: Limitations and caveats has {len(section.sentences)} "
                                 "sentences; keep at most 5, the ones most relevant to the question")
                         for i in range(MAX_LIMITATIONS, len(section.sentences))]
        for i, sentence in enumerate(section.sentences):
            unknown = sorted(set(sentence.finding_ids) - findings.keys())
            if unknown:
                problems.append((si, i, f"{UNKNOWN}: {unknown}"))
                continue
            cited = [findings[f] for f in sentence.finding_ids]
            if not cited and section.heading not in UNCITED_OK:
                problems.append((si, i, "makes a claim without citing a finding"))
            if cited:
                ceiling = min((CEILING[f.confidence] for f in cited), key=HEDGE_RANK.__getitem__)
                if HEDGE_RANK[sentence.hedge] > HEDGE_RANK[ceiling]:
                    problems.append((si, i, f"worded as '{sentence.hedge}' but the cited evidence only "
                                            f"supports '{ceiling}'"))
            if any(f.agent != "literature" for f in cited) and CAUSAL.search(sentence.text):
                problems.append((si, i, "causal wording about tract-level data associations"))
            if section.heading == "What the literature says" and any(f.agent != "literature" for f in cited):
                problems.append((si, i, "the literature section cites a data finding"))
            if section.heading == "What our data show" and any(f.agent in {"literature", "reconciler"} for f in cited):
                problems.append((si, i, "the data section cites a literature or comparison finding"))
            if section.heading == "Where they agree or disagree" and any(f.agent != "reconciler" for f in cited):
                problems.append((si, i, "the agree/disagree section cites only comparison (reconciler) findings"))
    return problems


def check_report(report: Report, findings: dict[str, Finding]) -> list[str]:
    """Rule violations in plain English, used as revision feedback for the Writer."""
    return [f"{where(report, si, i)}: {message}" for si, i, message in check_report_items(report, findings)]


def writer_item(f: Finding) -> dict:
    """What the Writer sees per finding: the wording ceiling is given, not left for the model to look up."""
    item = {"id": f.id, "agent": f.agent, "claim": f.claim, "confidence": f.confidence,
            "strongest_wording": CEILING[f.confidence]}
    if f.agent == "reconciler":
        item["verdict"] = f.evidence["verdict"]
    ci = f.evidence.get("ci95") if f.agent != "reconciler" else None
    if ci and "estimate" in f.evidence:
        item["interval_includes_zero"] = ci[0] <= 0 <= ci[1]
    return item


def writer_prompt(question: str, results: list[AgentResult], feedback: list[str] | None = None) -> str:
    findings, _, gaps = pool(results)
    items = [writer_item(f) for f in findings.values()]
    caveats: dict[str, list[str]] = {}  # caveats repeat across findings; list each once
    for f in findings.values():
        for caveat in f.caveats:
            if not caveat.startswith(PROVENANCE):
                caveats.setdefault(caveat, []).append(f.id)
    text = (f"Research question: {question}\n\nFindings (JSON):\n{json.dumps(items, indent=1)}\n\n"
            f"Caveats and the findings they apply to (JSON):\n"
            f"{json.dumps([{'caveat': c, 'finding_ids': ids} for c, ids in caveats.items()], indent=1)}")
    if not any(f.agent == "literature" for f in findings.values()):
        text += "\n\nNo literature findings were provided."
    if not any(f.agent == "reconciler" for f in findings.values()):
        text += "\n\nNo comparison findings were provided; omit the agree/disagree section."
    if gaps:
        text += ("\n\nKnown gaps (already listed under Open gaps; do not repeat them in Report.gaps):\n"
                 + "\n".join(f"- {g}" for g in gaps))
    if feedback:
        text += "\n\nYour previous draft had these problems. Fix every one:\n" + "\n".join(f"- {p}" for p in feedback)
    return text


def format_source(source: Source | PaperSource) -> str:
    if isinstance(source, PaperSource):
        authors = ", ".join(source.authors[:3]) + (" et al." if len(source.authors) > 3 else "")
        link = f"https://doi.org/{source.doi}" if source.doi else source.url
        tag = " [Preprint]" if source.preprint else ""
        return f"{authors or 'Unknown authors'} ({source.year or 'n.d.'}). {source.title}. *{source.venue or source.database}*.{tag} {link}"
    return f"{source.title}. {source.release}. {source.url} (retrieved {source.retrieved_at[:10]})."


def render(report: Report, results: list[AgentResult], *, extra_gaps: list[str] = (),
           verification: list[str] = ()) -> str:
    """Markdown with numbered source citations, Open gaps from every agent, Verification notes, and References."""
    findings, sources, gaps = pool(results)
    numbers: dict[str, int] = {}

    def cite(finding_ids: list[str]) -> str:
        ids = list(dict.fromkeys(s for f in finding_ids for s in findings[f].source_ids if s in sources))
        for source_id in ids:
            numbers.setdefault(source_id, len(numbers) + 1)
        return f" [{', '.join(str(numbers[s]) for s in ids)}]" if ids else ""

    blocks = []
    for section in report.sections:
        if section.sentences:
            body = " ".join(_cited(s.text, cite(s.finding_ids)) for s in section.sentences)
            blocks += [f"## {section.heading}", body]
    blocks.append("## Open gaps")
    all_gaps = list(dict.fromkeys(gaps + report.gaps + list(extra_gaps)))
    blocks.append("\n".join(f"- {g}" for g in all_gaps) or "- None reported.")
    if verification:
        blocks += ["## Verification", "\n".join(f"- {note}" for note in verification)]
    if numbers:
        blocks.append("## References")
        blocks.append("\n".join(f"{n}. {format_source(sources[s])}" for s, n in numbers.items()))
    return "\n\n".join(blocks)


def _cited(text: str, citation: str) -> str:
    """Place the citation before the sentence's final punctuation."""
    text = text.strip()
    if citation and text and text[-1] in ".!?":
        return text[:-1] + citation + text[-1]
    return text + citation
