"""Record real Crossref/PubMed responses for tests/test_verify.py. Network required; no keys used.

    uv run python -m scripts.record_verify_responses
"""
import asyncio
import json

import httpx

from data import ROOT, PaperSource
from verify import check_citations

OUT = ROOT / "tests/verify_responses.json"


def paper(id, title, year, doi=None, pmid=None):
    return PaperSource(id=id, doi=doi, pmid=pmid, title=title, authors=[], year=year, url="https://example.org",
                       database="pubmed", retrieved_at="2026-11-09")


SOURCES = [
    paper("valid", "Neighborhood walkability and physical activity among older women: Tests of mediation by "
          "environmental perceptions and moderation by depressive symptoms", 2018,
          doi="10.1016/j.ypmed.2018.08.008", pmid="30092314"),
    paper("retracted", "Ileal-lymphoid-nodular hyperplasia, non-specific colitis, and pervasive developmental "
          "disorder in children", 1998, doi="10.1016/S0140-6736(97)11096-0", pmid="9500320"),
    paper("missing", "A study that does not exist", 2020, doi="10.9999/sidewalk-psych-does-not-exist",
          pmid="99999999999"),
]


async def main():
    recorded = {}

    async def keep(response: httpx.Response):
        await response.aread()
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
        if body and isinstance(body.get("message"), dict):
            body["message"].pop("reference", None)  # long reference lists are not used; keep the file small
        recorded[str(response.request.url)] = {"status": response.status_code, "json": body,
                                               "text": None if body is not None else response.text[:200]}

    async with httpx.AsyncClient(timeout=20, follow_redirects=True, event_hooks={"response": [keep]}) as client:
        checks = await check_citations(SOURCES, client)
    for check in checks:
        print(check.source_id, check.status, check.details)
    OUT.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {len(recorded)} responses to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
