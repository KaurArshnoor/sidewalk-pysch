"""Record real PubMed/OpenAlex/PsyArXiv search responses for tests/test_literature.py. Network required.

    uv run python -m scripts.record_literature_responses

No keys are used (OPENALEX_API_KEY is ignored here so recorded URLs match the tests).
"""
import asyncio
import json
import os
import re

import httpx

from data import ROOT
from literature import LiteratureContext

OUT = ROOT / "tests/literature_responses.json"
SEARCHES = [("search_pubmed", ["walkability", "mental distress"]),
            ("search_openalex", ["walkability", "loneliness"]),
            ("search_psyarxiv", ["loneliness", "neighborhood"])]


OSF_FIELDS = ["title", "description", "date_published", "date_withdrawn", "has_prereg_links", "doi"]


def trim_osf(item: dict) -> dict:
    """Only the fields parse_osf reads (OSF embeds full contributor profiles)."""
    names = [{"embeds": {"users": {"data": {"attributes": {"full_name": name}}}}}
             for c in item.get("embeds", {}).get("contributors", {}).get("data", [])
             if (name := c.get("embeds", {}).get("users", {}).get("data", {}).get("attributes", {}).get("full_name"))]
    return {"id": item["id"], "links": item.get("links", {}),
            "attributes": {k: item["attributes"].get(k) for k in OSF_FIELDS},
            "embeds": {"contributors": {"data": names}}}


async def main():
    os.environ.pop("OPENALEX_API_KEY", None)
    recorded = {}

    async def keep(response: httpx.Response):
        await response.aread()
        is_json = response.headers.get("content-type", "").split(";")[0].endswith("json")
        body, text = (response.json(), None) if is_json else (None, response.text)
        if body and "osf.io" in str(response.request.url):
            body = {"data": [trim_osf(item) for item in body.get("data", [])]}
        if text:  # PubMed sections parse_pubmed does not read; keep the file small
            text = re.sub(r"<(ReferenceList|MeshHeadingList|KeywordList|GrantList|History|ChemicalList|"
                          r"CommentsCorrectionsList|AffiliationInfo)>.*?</\1>", "", text, flags=re.DOTALL)
        recorded[str(response.request.url)] = {"status": response.status_code, "json": body, "text": text}

    async with httpx.AsyncClient(timeout=20, follow_redirects=True, event_hooks={"response": [keep]}) as client:
        ctx = LiteratureContext(client=client)
        for name, concepts in SEARCHES:
            result = await ctx.invoke(name, concepts=concepts)
            print(name, result["status"], len(result.get("papers", [])), "papers")
    if any(r["status"] != 200 for r in recorded.values()):
        raise SystemExit(f"Not saved: a request failed {[(u, r['status']) for u, r in recorded.items() if r['status'] != 200]}")
    OUT.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {len(recorded)} responses ({OUT.stat().st_size // 1024} KB) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
