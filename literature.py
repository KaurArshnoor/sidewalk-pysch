"""Literature search, study extraction checks, and the evidence-strength rubric. No model calls.

The literature agent (app.py) searches PubMed, OpenAlex and PsyArXiv through these functions, reads
abstracts, and records studies with record_study. Python checks every extraction against the abstract
it came from and scores evidence strength; the model never supplies an unchecked fact.
API limits are documented in docs/external-apis.md.
"""
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel

import apis
from data import Effect, EvidenceStrength, Finding, PaperSource, StudyEvidence, ToolCall
from verify import numbers_in

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OPENALEX = "https://api.openalex.org/works"
OSF = "https://api.osf.io/v2/preprints/"
PER_SEARCH = 8  # papers per database per search
ABSTRACT_CHARS = 2000  # shown to the model; the full abstract is kept for quote checks
MAX_CONCEPTS = 3
OSF_TERMS = 5  # PsyArXiv title searches per call (one request each)

# Query expansion. Keys are concepts the agent can name; unknown concepts are searched as given.
SYNONYMS = {
    "walkability": ["walkability", "walkable neighborhood", "neighborhood walkability", "street connectivity",
                    "built environment"],
    "mental distress": ["psychological distress", "mental distress", "depression", "depressive symptoms",
                        "anxiety", "mental health"],
    "depression": ["depression", "depressive symptoms", "depressive disorder"],
    "loneliness": ["loneliness", "social isolation", "social connectedness", "social cohesion"],
    "social connection": ["social connection", "social connectedness", "social support", "social cohesion",
                          "social isolation"],
    "green space": ["green space", "greenspace", "park access", "urban parks", "urban green"],
    "amenities": ["neighborhood amenities", "third places", "gathering places", "community facilities"],
}


# Queries

def expand(concepts: list[str]) -> list[list[str]]:
    """One synonym group per concept (at most three). Quotes and commas are removed: they break queries."""
    groups = []
    for concept in concepts[:MAX_CONCEPTS]:
        key = " ".join(str(concept).lower().split())
        terms = SYNONYMS.get(key, [key])
        terms = [" ".join(re.sub(r"[\",()\[\]]", " ", t).split()) for t in terms]
        if any(terms):
            groups.append(list(dict.fromkeys(t for t in terms if t)))
    return groups


def pubmed_query(groups: list[list[str]]) -> str:
    return " AND ".join("(" + " OR ".join(f'"{t}"[tiab]' for t in g) + ")" for g in groups)


def openalex_query(groups: list[list[str]]) -> str:
    return " AND ".join("(" + " OR ".join(f'"{t}"' for t in g) + ")" for g in groups)


# Papers

class Paper(BaseModel):
    id: str
    doi: str | None = None
    pmid: str | None = None
    title: str
    authors: list[str]
    year: int | None = None
    venue: str | None = None
    url: str
    database: str
    preprint: bool = False
    abstract: str
    publication_types: list[str] = []
    preregistered: bool | None = None

    def summary(self) -> dict:
        abstract = self.abstract if len(self.abstract) <= ABSTRACT_CHARS else self.abstract[:ABSTRACT_CHARS] + " [...]"
        return {"paper_id": self.id, "title": self.title, "year": self.year, "venue": self.venue,
                "database": self.database, "preprint": self.preprint,
                "publication_types": self.publication_types, "abstract": abstract}

    def source(self) -> PaperSource:
        return PaperSource(id=self.id, doi=self.doi, pmid=self.pmid, title=self.title, authors=self.authors,
                           year=self.year, venue=self.venue, url=self.url, database=self.database,
                           preprint=self.preprint, retrieved_at=datetime.now(timezone.utc).isoformat())


def paper_id(doi: str | None, pmid: str | None, fallback: str) -> str:
    return f"doi:{doi.lower()}" if doi else f"pmid:{pmid}" if pmid else fallback


# Registry identifiers; a bare mention of "preregistration" is not enough ("was not preregistered").
REGISTRY = re.compile(r"\b(CRD42\d{6,}|NCT\d{8}|ISRCTN\d{8}|ACTRN\d{14}|osf\.io/[a-z0-9]{5}\b|aspredicted\.org|"
                      r"registered (?:in|with|at|on) (?:PROSPERO|ClinicalTrials\.gov|the Open Science Framework))",
                      re.IGNORECASE)


def preregistered(text: str) -> bool | None:
    """True when a registry identifier appears; None (unknown) otherwise, never False from text alone."""
    return True if REGISTRY.search(text) else None


def _text(element) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def parse_pubmed(xml: str) -> list[Paper]:
    papers = []
    for article in ET.fromstring(xml).iter("PubmedArticle"):
        citation = article.find("MedlineCitation")
        a = citation.find("Article")
        pmid = citation.findtext("PMID")
        parts = []
        for block in a.findall("Abstract/AbstractText"):
            label = block.get("Label")
            parts.append(f"{label}: {_text(block)}" if label else _text(block))
        abstract = " ".join(p for p in parts if p)
        if not abstract:
            continue
        authors = []
        for author in a.findall("AuthorList/Author"):
            if author.findtext("LastName"):
                authors.append(" ".join(filter(None, [author.findtext("LastName"), author.findtext("Initials")])))
            elif author.findtext("CollectiveName"):
                authors.append(author.findtext("CollectiveName"))
        date = a.find("Journal/JournalIssue/PubDate")
        year = re.search(r"\d{4}", (date.findtext("Year") or date.findtext("MedlineDate") or "") if date is not None else "")
        doi = next((e.text for e in a.findall("ELocationID") if e.get("EIdType") == "doi" and e.text), None) or \
            next((e.text for e in article.findall("PubmedData/ArticleIdList/ArticleId")
                  if e.get("IdType") == "doi" and e.text), None)
        types = [t.text for t in a.findall("PublicationTypeList/PublicationType") if t.text]
        title = _text(a.find("ArticleTitle"))
        papers.append(Paper(id=paper_id(doi, pmid, f"pmid:{pmid}"), doi=doi, pmid=pmid, title=title,
                            authors=authors, year=int(year.group()) if year else None,
                            venue=a.findtext("Journal/Title"), url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            database="pubmed", abstract=abstract, publication_types=types,
                            preregistered=preregistered(abstract)))
    return papers


def parse_openalex(data: dict) -> list[Paper]:
    papers = []
    for work in data.get("results", []):
        index = work.get("abstract_inverted_index") or {}
        if not index or not work.get("title"):
            continue
        abstract = " ".join(word for _, word in sorted((p, w) for w, positions in index.items() for p in positions))
        doi = (work.get("doi") or "").removeprefix("https://doi.org/") or None
        pmid = ((work.get("ids") or {}).get("pmid") or "").rstrip("/").rsplit("/", 1)[-1] or None
        source = (work.get("primary_location") or {}).get("source") or {}
        papers.append(Paper(id=paper_id(doi, pmid, work["id"]), doi=doi, pmid=pmid, title=work["title"],
                            authors=[a["author"]["display_name"] for a in work.get("authorships", [])
                                     if a.get("author", {}).get("display_name")],
                            year=work.get("publication_year"), venue=source.get("display_name"),
                            url=work.get("doi") or work["id"], database="openalex",
                            preprint=work.get("type") == "preprint", abstract=abstract,
                            publication_types=[work["type"]] if work.get("type") else [],
                            preregistered=preregistered(abstract)))
    return papers


def parse_osf(data: dict) -> list[Paper]:
    papers = []
    for item in data.get("data", []):
        a = item.get("attributes", {})
        abstract = " ".join((a.get("description") or "").split())
        if not abstract or a.get("date_withdrawn") or not a.get("title"):
            continue
        links = item.get("links", {})
        doi = (links.get("preprint_doi") or "").removeprefix("https://doi.org/") or None
        contributors = item.get("embeds", {}).get("contributors", {}).get("data", [])
        authors = [c.get("embeds", {}).get("users", {}).get("data", {}).get("attributes", {}).get("full_name")
                   for c in contributors]
        flag = {"available": True, "no": False}.get(a.get("has_prereg_links"))
        published = a.get("date_published") or ""
        papers.append(Paper(id=paper_id(doi, None, f"osf:{item['id']}"), doi=doi, title=" ".join(a["title"].split()),
                            authors=[x for x in authors if x], year=int(published[:4]) if published[:4].isdigit() else None,
                            venue="PsyArXiv", url=links.get("html") or f"https://osf.io/{item['id']}",
                            database="psyarxiv", preprint=True, abstract=abstract, publication_types=["preprint"],
                            preregistered=flag if flag is not None else preregistered(abstract)))
    return papers


# Searches: each returns (query used, papers)

async def search_pubmed(client: httpx.AsyncClient, limits: dict, groups: list[list[str]]):
    query = pubmed_query(groups)
    found = await apis.get(client, ESEARCH, limits["pubmed"], db="pubmed", term=query, retmax=PER_SEARCH,
                           sort="relevance", retmode="json")
    ids = (found or {}).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return query, []
    xml = await apis.get(client, EFETCH, limits["pubmed"], as_text=True, db="pubmed", id=",".join(ids),
                         retmode="xml")
    return query, parse_pubmed(xml or "<PubmedArticleSet/>")


async def search_openalex(client: httpx.AsyncClient, limits: dict, groups: list[list[str]]):
    query = openalex_query(groups)
    params = {"filter": f"title_and_abstract.search:{query},has_abstract:true", "per-page": PER_SEARCH,
              "select": "id,doi,title,publication_year,type,authorships,primary_location,ids,abstract_inverted_index"}
    if os.getenv("OPENALEX_API_KEY"):  # optional: raises the shared anonymous daily budget
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    data = await apis.get(client, OPENALEX, limits["openalex"], **params)
    return query, parse_openalex(data or {})


async def search_psyarxiv(client: httpx.AsyncClient, limits: dict, groups: list[list[str]]):
    """OSF searches titles only, one term per request: search the first concept's terms in titles, then
    keep preprints whose title or abstract also mentions every other concept."""
    terms = groups[0][:OSF_TERMS]
    papers: dict[str, tuple[int, Paper]] = {}  # base preprint id → (version, paper)
    for term in terms:
        data = await apis.get(client, OSF, limits["osf"], **{
            "filter[provider]": "psyarxiv", "filter[title][icontains]": term, "page[size]": 10,
            "embed": "contributors"})
        for paper in parse_osf(data or {}):
            text = f"{paper.title} {paper.abstract}".lower()
            if all(any(t.lower() in text for t in group) for group in groups[1:]):
                # Each version is listed separately (purme, purme_v2, purme_v3); keep the highest.
                base, version = re.fullmatch(r"(.*?)(?:_v(\d+))?", paper.id).groups()
                version = int(version or 0)
                if base not in papers or version > papers[base][0]:
                    papers[base] = (version, paper)
    papers = {base: paper for base, (_, paper) in papers.items()}
    query = f"title contains any of {terms}" + (f"; title or abstract mentions {groups[1:]}" if groups[1:] else "")
    return query, list(papers.values())[:PER_SEARCH]


SEARCHES = {"search_pubmed": search_pubmed, "search_openalex": search_openalex, "search_psyarxiv": search_psyarxiv}


# Evidence-strength rubric

DESIGN_POINTS = {"meta_analysis": 3, "rct": 3, "longitudinal": 2, "cross_sectional": 1,
                 "ecological": 0, "qualitative": 0, "other": 0}
DESIGN_LABEL = {"meta_analysis": "meta-analysis", "rct": "randomized trial", "longitudinal": "longitudinal",
                "cross_sectional": "cross-sectional", "ecological": "ecological", "qualitative": "qualitative",
                "other": "other-design"}


def score(design: str, n: int | None, prereg: bool | None, preprint: bool) -> EvidenceStrength:
    points, reasons = DESIGN_POINTS[design], [f"{DESIGN_LABEL[design]} design +{DESIGN_POINTS[design]}"]
    if n is not None and n >= 1000:
        points += 1
        reasons.append("n ≥ 1,000 +1")
    if prereg:
        points += 1
        reasons.append("preregistered +1")
    if preprint:
        points -= 1
        reasons.append("preprint (not peer reviewed) −1")
    tier = "strong" if points >= 4 else "moderate" if points >= 2 else "weak"
    return EvidenceStrength(score=points, tier=tier, reasons=reasons)


# A stronger design must be visible in the abstract, title or publication type.
DESIGN_EVIDENCE = {
    "meta_analysis": re.compile(r"meta-?analy", re.IGNORECASE),
    "rct": re.compile(r"randomi[sz]ed|randomly (?:assigned|allocated)", re.IGNORECASE),
    "longitudinal": re.compile(r"longitudinal|cohort|prospective|follow-?up|panel stud|\bwaves?\b", re.IGNORECASE),
}


def _clean(text: str) -> str:
    """Whitespace, dash and quote variants normalized for quote matching."""
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', text))
    return " ".join(text.split())


def _in_quote(value: float, quote_numbers: list[float]) -> bool:
    return any(abs(value - q) < 1e-9 for q in quote_numbers)


# The agent's context

@dataclass
class LiteratureContext:
    client: httpx.AsyncClient | None = None  # tests inject a recorded transport
    role: str = "literature"
    papers: dict[str, Paper] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)  # doi/pmid/id → paper id
    findings: dict[str, Finding] = field(default_factory=dict)
    recorded: dict[str, str] = field(default_factory=dict)  # paper id → finding id
    calls: list[ToolCall] = field(default_factory=list)
    max_calls: int = 25
    reviewing_abstracts: bool = False
    _limits: dict | None = None

    @property
    def sources(self) -> list[PaperSource]:
        return [self.papers[p].source() for p in self.recorded]

    async def invoke(self, name: str, **arguments) -> dict:
        if len(self.calls) >= self.max_calls:
            raise ValueError("Tool budget exhausted")
        start = time.perf_counter()
        try:
            result = await self._invoke(name, **arguments)
        except (ValueError, KeyError) as exc:
            result = {"status": "error", "reason": str(exc)}
        self.calls.append(ToolCall(agent=self.role, name=name, arguments=arguments, result=result,
                                   duration_ms=round((time.perf_counter() - start) * 1000, 2)))
        return result

    async def _invoke(self, name: str, **arguments) -> dict:
        if name in SEARCHES:
            return await self._search(name, arguments.get("concepts") or [])
        if name == "record_study":
            return self._record(**arguments)
        raise ValueError(f"Unknown tool: {name}")

    async def _search(self, name: str, concepts: list[str]) -> dict:
        database = name.removeprefix("search_")
        groups = expand(concepts)
        if not groups:
            raise ValueError("Give one to three concepts, for example ['walkability', 'loneliness']")
        self._limits = self._limits or apis.limits()
        try:
            if self.client is not None:
                query, papers = await SEARCHES[name](self.client, self._limits, groups)
            else:
                async with apis.new_client() as client:
                    query, papers = await SEARCHES[name](client, self._limits, groups)
        except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
            return {"status": "unavailable", "database": database, "reason": type(exc).__name__,
                    "note": f"{database} could not be searched; record this as a gap and use the other databases."}
        new, seen = [], 0
        for paper in papers:
            if self._resolve(paper.id) or (paper.doi and self._resolve(paper.doi)) or (paper.pmid and self._resolve(paper.pmid)):
                seen += 1
                continue
            self.papers[paper.id] = paper
            for key in filter(None, [paper.id, paper.doi and paper.doi.lower(), paper.pmid]):
                self.aliases[key.lower()] = paper.id
            new.append(paper)
        return {"status": "ok", "database": database, "query": query, "expanded_concepts": groups,
                "already_seen": seen, "papers": [p.summary() for p in new]}

    def _resolve(self, key: str) -> str | None:
        key = key.strip().lower().removeprefix("https://doi.org/")
        return self.aliases.get(key) or self.aliases.get(f"doi:{key}") or self.aliases.get(f"pmid:{key}")

    def _record(self, paper_id: str, design: str, quote: str, exposure: str, outcome: str, n: int | None = None,
                effect_metric: str | None = None, effect_value: float | None = None,
                ci_low: float | None = None, ci_high: float | None = None) -> dict:
        pid = self._resolve(paper_id)
        if pid is None:
            raise ValueError("Unknown paper_id: record only papers returned by a search in this run")
        if pid in self.recorded:
            raise ValueError(f"{pid} is already recorded as {self.recorded[pid]}")
        if design not in DESIGN_POINTS:
            raise ValueError(f"design must be one of {sorted(DESIGN_POINTS)}")
        paper = self.papers[pid]
        quote = _clean(quote).strip(' "')
        if len(quote) < 20 or quote not in _clean(paper.abstract):
            raise ValueError("quote must be copied exactly from this paper's abstract (at least 20 characters)")
        exposure, outcome = " ".join(exposure.split())[:150], " ".join(outcome.split())[:150]
        if not exposure or not outcome:
            raise ValueError("State the exposure and outcome the study measured")
        pattern = DESIGN_EVIDENCE.get(design)
        evidence_text = " ".join([paper.title, paper.abstract] + paper.publication_types)
        if pattern and not pattern.search(evidence_text):
            raise ValueError(f"The abstract does not describe a {DESIGN_LABEL[design]} design; choose the design it states")
        # The sample size is usually in a methods sentence, not the result sentence being quoted, so n is checked
        # against the whole abstract. The effect and its interval must be in the quote: the quote is their evidence.
        if n is not None and not _in_quote(n, [v for _, v in numbers_in(_clean(paper.abstract))]):
            raise ValueError(f"n={n} does not appear in this paper's abstract; omit n and record the study without it")
        numbers = [v for _, v in numbers_in(quote)]
        for label, value in [("effect_value", effect_value), ("ci_low", ci_low), ("ci_high", ci_high)]:
            if value is not None and not _in_quote(value, numbers):
                raise ValueError(f"{label}={value} does not appear in the quote; quote the sentence that reports it, "
                                 "or record the study without the effect")
        if effect_value is not None and not effect_metric:
            raise ValueError("effect_metric is required with effect_value (for example OR, beta, r)")
        if (ci_low is None) != (ci_high is None) or (ci_low is not None and effect_value is None):
            raise ValueError("Give both ci_low and ci_high, together with effect_value")
        if ci_low is not None and not ci_low <= effect_value <= ci_high:
            raise ValueError("The confidence interval must contain the effect estimate")
        effect = Effect(metric=effect_metric, value=effect_value,
                        ci95=(ci_low, ci_high) if ci_low is not None else None) if effect_value is not None else None
        strength = score(design, n, paper.preregistered, paper.preprint)
        evidence = StudyEvidence(paper_id=pid, design=design, exposure=exposure, outcome=outcome, n=n, effect=effect,
                                 preregistered=paper.preregistered, quote=quote, strength=strength)

        who = f"{paper.authors[0].split()[0] if paper.authors else 'Unknown'}{' et al.' if len(paper.authors) > 1 else ''}"
        details = "; ".join(filter(None, [f"{who} {paper.year or 'n.d.'}", f"n={n:,}" if n is not None else None]))
        result = ""
        if effect:
            result = f" reported {effect.metric} {effect.value:g}"
            if effect.ci95:
                result += f" (95% CI {effect.ci95[0]:g} to {effect.ci95[1]:g})"
        claim = (f"A {DESIGN_LABEL[design]} study ({details}) of {exposure} and {outcome}{result}. "
                 f"Abstract: \"{quote}\"")
        caveats = ["Extracted from the abstract only; the full text was not reviewed."]
        if paper.preprint:
            caveats.append("Preprint: not peer reviewed.")
        if design in {"cross_sectional", "ecological"}:
            caveats.append(f"{DESIGN_LABEL[design].capitalize()} design: cannot establish direction, causation, "
                           "or individual-level effects.")
        if not re.search(r"lonel|social|isolat", outcome, re.IGNORECASE):
            caveats.append(f"Measures {outcome}, not loneliness or social connection.")
        if paper.preregistered is None:
            caveats.append("Preregistration status is not reported in the abstract.")

        body = {"agent": "literature", "claim": claim, "evidence": evidence.model_dump(), "source_ids": [pid],
                "confidence": f"literature_{strength.tier}", "caveats": caveats}
        identifier = "f-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        finding = Finding(id=identifier, **body)
        self.findings[identifier] = finding
        self.recorded[pid] = identifier
        return finding.model_dump()
