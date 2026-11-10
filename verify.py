"""Verifier checks: citations (Crossref, PubMed), numbers, and the format for claim-by-claim verdicts.

No model calls; app.py runs the Verifier agent. Public APIs only: no keys or contact details are sent.
Crossref's work record includes Retraction Watch updates under "updated-by"; PubMed tags retracted
articles with the publication type "Retracted Publication".
"""
import asyncio
import difflib
import json
import re
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from apis import get, limits as api_limits, new_client
from data import AgentResult, Finding, PaperSource, Source
from report import PROVENANCE, Report, check_report_items, where

CROSSREF = "https://api.crossref.org/works/"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

Status = Literal["ok", "not_found", "metadata_mismatch", "retracted", "expression_of_concern",
                 "corrected", "unchecked", "lookup_failed"]
EXCLUDE = {"retracted", "not_found"}  # findings citing these sources are removed before writing


class CitationCheck(BaseModel):
    source_id: str
    status: Status
    details: str


class Verdict(BaseModel):
    sentence: int  # the number shown in the verifier prompt
    status: Literal["supported", "overstated", "unsupported", "misattributed"]
    reason: str


class VerifierOutput(BaseModel):
    """The Verifier agent's output_type."""

    verdicts: list[Verdict]


# Citations

def _normalize(title: str) -> str:
    title = re.sub(r"^\s*(retracted|withdrawn)\s*[:\-–—]\s*", "", title, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[^\w\s]", " ", title.lower()).split())


def _same_title(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= 0.9


async def crossref_record(client: httpx.AsyncClient, doi: str, limit: asyncio.Semaphore) -> dict | None:
    data = await get(client, CROSSREF + quote(doi, safe="/()"), limit)
    return data["message"] if data else None


async def pubmed_record(client: httpx.AsyncClient, pmid: str, limit: asyncio.Semaphore) -> dict | None:
    data = await get(client, ESUMMARY, limit, db="pubmed", id=pmid, retmode="json")
    record = (data or {}).get("result", {}).get(pmid)
    return None if not record or "error" in record else record


def _crossref_year(record: dict) -> int | None:
    for key in ("published", "issued", "published-print", "published-online"):
        parts = record.get(key, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def assess(source: PaperSource, crossref: dict | None, pubmed: dict | None,
           failed: frozenset[str] = frozenset()) -> CitationCheck:
    """Compare looked-up records with the cited source. Records are None when not found;
    failed names identifiers ("DOI", "PMID") whose lookup errored, which are left out of the comparison."""
    def check(status: Status, details: str) -> CitationCheck:
        return CitationCheck(source_id=source.id, status=status, details=details)

    identifiers = [name for name, value in (("DOI", source.doi), ("PMID", source.pmid))
                   if value and name not in failed]
    found = {name: record for name, record in (("DOI", crossref), ("PMID", pubmed))
             if name in identifiers and record is not None}
    if not found:
        return check("not_found", f"No record for {' or '.join(identifiers)}" if identifiers
                     else "The source has no DOI or PMID to check")

    updates = {u.get("type", "").lower() for u in (crossref or {}).get("updated-by", [])}
    titles = (crossref or {}).get("title") or []
    pubtypes = set((pubmed or {}).get("pubtype", []))
    retracted_title = any(re.match(r"\s*(retracted|withdrawn)\s*[:\-–—]", t, re.IGNORECASE) for t in titles)
    if "Retracted Publication" in pubtypes or updates & {"retraction", "withdrawal", "removal"} or retracted_title:
        return check("retracted", "Retracted according to " + ("PubMed" if "Retracted Publication" in pubtypes else "Crossref"))
    if "expression_of_concern" in updates or "Expression of Concern" in pubtypes:
        return check("expression_of_concern", "An expression of concern has been published")

    mismatches = [f"{name} not found" for name in identifiers if name not in found]
    looked_up = []
    if crossref is not None:
        looked_up.append(("Crossref", titles[0] if titles else "", _crossref_year(crossref)))
    if pubmed is not None:
        year = re.match(r"\d{4}", pubmed.get("pubdate", ""))
        looked_up.append(("PubMed", pubmed.get("title", ""), int(year.group()) if year else None))
        dois = [a["value"].lower() for a in pubmed.get("articleids", []) if a.get("idtype") == "doi"]
        if source.doi and dois and source.doi.lower() not in dois:
            mismatches.append(f"PubMed lists DOI {dois[0]}, not {source.doi}")
    for database, title, year in looked_up:
        if title and not _same_title(source.title, title):
            mismatches.append(f"{database} title is '{title}'")
        if source.year and year and abs(source.year - year) > 1:
            mismatches.append(f"{database} year is {year}, not {source.year}")
    if mismatches:
        return check("metadata_mismatch", "; ".join(mismatches))
    if updates & {"correction", "erratum", "corrigendum"}:
        return check("corrected", "A correction has been published; the study is kept")
    return check("ok", "Found in " + " and ".join(name for name, _, _ in looked_up))


async def check_source(client: httpx.AsyncClient, source: Source | PaperSource,
                       limits: dict[str, asyncio.Semaphore]) -> CitationCheck:
    if not isinstance(source, PaperSource):
        return CitationCheck(source_id=source.id, status="unchecked",
                             details="Dataset source; checksums verified when the data were built")
    lookups = {"DOI": (crossref_record, source.doi, limits["crossref"]),
               "PMID": (pubmed_record, source.pmid, limits["pubmed"])}
    records, failed, errors = {}, set(), []
    for name, (lookup, identifier, limit) in lookups.items():
        if not identifier:
            records[name] = None
            continue
        try:
            records[name] = await lookup(client, identifier, limit)
        except (httpx.HTTPError, ValueError) as exc:  # ValueError: response was not JSON
            records[name], failed = None, failed | {name}
            errors.append(f"{name} lookup failed ({type(exc).__name__})")
    if failed and not any(records.values()):
        return CitationCheck(source_id=source.id, status="lookup_failed",
                             details="; ".join(errors) + "; the study is kept but unverified")
    result = assess(source, records["DOI"], records["PMID"], frozenset(failed))
    if errors:
        result.details += " (" + "; ".join(errors) + ")"
    return result


async def check_citations(sources: list[Source | PaperSource],
                          client: httpx.AsyncClient | None = None) -> list[CitationCheck]:
    limits = api_limits()
    if client is not None:
        return list(await asyncio.gather(*(check_source(client, s, limits) for s in sources)))
    async with new_client() as own:
        return list(await asyncio.gather(*(check_source(own, s, limits) for s in sources)))


def exclude_bad_sources(results: list[AgentResult], checks: list[CitationCheck]):
    """Remove findings that cite retracted or missing papers. Returns (results, gaps)."""
    bad = {c.source_id: c for c in checks if c.status in EXCLUDE}
    titles = {s.id: s.title for r in results for s in r.sources}
    kept = []
    for result in results:
        findings = [f for f in result.findings if not bad.keys() & set(f.source_ids)]
        cited = {s for f in findings for s in f.source_ids}
        kept.append(result.model_copy(update={"findings": findings,
                                               "sources": [s for s in result.sources if s.id in cited]}))
    label = {"retracted": "retracted", "not_found": "unverifiable"}
    gaps = [f"Excluded {label[c.status]} study: {titles.get(c.source_id, c.source_id)} ({c.details})."
            for c in bad.values()]
    return kept, gaps


def citation_notes(checks: list[CitationCheck]) -> list[str]:
    papers = [c for c in checks if c.status != "unchecked"]
    if not papers:
        return ["No literature citations to check; dataset sources were verified when the data were built."]
    notes = [f"Checked {len(papers)} paper citation(s) against Crossref/PubMed; "
             f"{sum(c.status == 'ok' for c in papers)} matched."]
    notes += [f"{c.source_id}: {c.status.replace('_', ' ')}. {c.details}" for c in papers if c.status != "ok"]
    return notes


# Numbers

NUMBER = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")
IGNORED = {95, 100}  # "95% interval", "0–100 units"


def numbers_in(text: str) -> list[tuple[str, float]]:
    found = []
    for match in NUMBER.finditer(text):
        raw = match.group().rstrip(",").replace("−", "-").replace(",", "")
        try:
            found.append((match.group().rstrip(","), float(raw)))
        except ValueError:
            continue
    return found


def _ignored(raw: str, value: float) -> bool:
    whole = value.is_integer() and "." not in raw
    return whole and (abs(value) <= 10 or abs(value) in IGNORED or 1900 <= value <= 2100)


def _supports(value: float, raw: str, source: float) -> bool:
    """Exact match, or the source value rounded to the sentence's precision; sign may be stated in words."""
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return abs(round(abs(source), decimals) - abs(value)) < 10 ** -(decimals + 3)


def check_numbers(report: Report, findings: dict[str, Finding]) -> list[tuple[int, int, str]]:
    problems = []
    for si, section in enumerate(report.sections):
        for i, sentence in enumerate(section.sentences):
            cited = [findings[f] for f in sentence.finding_ids if f in findings]
            text = " ".join([f.claim for f in cited] + [c for f in cited for c in f.caveats]
                            + [json.dumps(f.evidence) for f in cited])
            available = [v for _, v in numbers_in(text)]
            missing = [raw for raw, value in numbers_in(sentence.text)
                       if not _ignored(raw, value) and not any(_supports(value, raw, s) for s in available)]
            if missing:
                problems.append((si, i, f"numbers not found in the cited findings: {', '.join(missing)}"))
    return problems


# Claim-by-claim verdicts

def sentences(report: Report) -> list[tuple[int, int]]:
    """(section index, sentence index) in reading order; the prompt numbers them from 1."""
    return [(si, i) for si, s in enumerate(report.sections) for i in range(len(s.sentences))]


def _compact(finding: Finding) -> dict:
    evidence = {k: v for k, v in finding.evidence.items() if len(json.dumps(v)) <= 1000}
    return {"id": finding.id, "agent": finding.agent, "claim": finding.claim, "confidence": finding.confidence,
            "evidence": evidence, "caveats": [c for c in finding.caveats if not c.startswith(PROVENANCE)]}


def verifier_prompt(report: Report, findings: dict[str, Finding]) -> str:
    items = []
    for n, (si, i) in enumerate(sentences(report), start=1):
        sentence = report.sections[si].sentences[i]
        items.append({"sentence": n, "section": report.sections[si].heading, "text": sentence.text,
                      "hedge": sentence.hedge,
                      "cited_findings": [_compact(findings[f]) for f in sentence.finding_ids if f in findings]})
    return "Judge each sentence against only its cited findings (JSON):\n" + json.dumps(items, indent=1)


def verdict_items(report: Report, output: VerifierOutput) -> list[tuple[int, int, str]]:
    """Problems from verdicts; a sentence without a verdict is a problem, never assumed supported."""
    positions = sentences(report)
    given = {}
    for verdict in output.verdicts:
        if 1 <= verdict.sentence <= len(positions):
            given.setdefault(verdict.sentence, verdict)
    problems = []
    for n, (si, i) in enumerate(positions, start=1):
        verdict = given.get(n)
        if verdict is None:
            problems.append((si, i, "the verifier returned no verdict"))
        elif verdict.status != "supported":
            problems.append((si, i, f"{verdict.status}: {verdict.reason}"))
    return problems


def feedback(report: Report, items: list[tuple[int, int, str]]) -> list[str]:
    return [f"{where(report, si, i)}: {message}" for si, i, message in items]


def all_problems(report: Report, findings: dict[str, Finding], output: VerifierOutput) -> list[tuple[int, int, str]]:
    return check_report_items(report, findings) + check_numbers(report, findings) + verdict_items(report, output)


def remove_flagged(report: Report, items: list[tuple[int, int, str]]) -> tuple[Report, list[str]]:
    """Drop every flagged sentence; return the cleaned report and a gap per removal."""
    flagged: dict[tuple[int, int], list[str]] = {}
    for si, i, message in items:
        flagged.setdefault((si, i), []).append(message)
    sections, gaps = [], []
    for si, section in enumerate(report.sections):
        kept = []
        for i, sentence in enumerate(section.sentences):
            if (si, i) in flagged:
                text = sentence.text if len(sentence.text) <= 120 else sentence.text[:117] + "..."
                gaps.append(f"Removed an unverified statement from {section.heading}: \"{text}\" "
                            f"({'; '.join(flagged[(si, i)])})")
            else:
                kept.append(sentence)
        if kept:
            sections.append(section.model_copy(update={"sentences": kept}))
    return report.model_copy(update={"sections": sections}), gaps
