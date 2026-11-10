"""Shared, polite HTTP access to public research APIs, used by the Verifier and the literature agent.

No keys or contact details are required. Limits for every service are documented in docs/external-apis.md.
"""
import asyncio

import httpx

HEADERS = {"User-Agent": "sidewalk-psych/0.1 (research coursework)"}
RETRIES = 3
TIMEOUT = 20

# Concurrent requests per service. Anonymous Crossref returns 429 above one; PubMed allows 3/second.
CONCURRENCY = {"crossref": 1, "pubmed": 2, "openalex": 2, "osf": 2}


def limits() -> dict[str, asyncio.Semaphore]:
    """Fresh per-run limits (semaphores belong to one event loop)."""
    return {name: asyncio.Semaphore(n) for name, n in CONCURRENCY.items()}


async def get(client: httpx.AsyncClient, url: str, limit: asyncio.Semaphore, *, as_text: bool = False,
              **params) -> dict | str | None:
    """JSON (or text) response, or None for 404. Retries 429 with backoff; other errors propagate."""
    for attempt in range(RETRIES + 1):
        async with limit:
            response = await client.get(url, params=params or None, headers=HEADERS)
            if response.status_code == 429 and attempt < RETRIES:
                retry_after = response.headers.get("Retry-After", "")
                await asyncio.sleep(min(float(retry_after), 10) if retry_after.isdigit() else 1.5 * (attempt + 1))
                continue
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.text if as_text else response.json()


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)
