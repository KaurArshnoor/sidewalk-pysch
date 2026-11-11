# External APIs used and their limits and additional details

Most agents run **without API keys**. This page lists each of the outside services that the different agents call, their limits when used anonymously (without a key), and what happens when any limit is reached. These limits are as of November 2026, and they might change in the future.

## Summary by agent

| Agent | Services called while answering | Key needed | Main risk |
|---|---|---|---|
| Built environment, Health/demographics, Analyst | None. Data are prebuilt in `data/processed/study` | No | None at runtime |
| **Literature researcher** | PubMed, OpenAlex, OSF (PsyArXiv) | No (OpenAlex key optional) | OpenAlex daily budget shared by IP |
| Writer | None | No | None |
| **Verifier** | Crossref, PubMed | No | Crossref is slow without a contact email address |

All agents also call the LLM provider set by `SIDEWALK_MODEL` (Vertex AI by default). Its quotas and
costs depend on the given Google Cloud project configurations and are not documented here.

The data-related agents only reach the network when the dataset is **rebuilt** (`scripts/build_data.py`,
`scripts/build_environment.py`), which is an optional process. Look at [Data build sources](#data-build-sources).

## Per service

### PubMed (NCBI E-utilities)
Used by: literature researcher (`esearch`, `efetch`), Verifier (`esummary`).

- **Without a key:** 3 requests per second per IP.
- **Our usage:** about 2 requests per literature search and 1 per checked PMID. We send at most 2 at a time.
- **At the limit:** HTTP 429. `apis.get` waits and retries up to 3 times.
- **If it still fails:** the literature search returns `"unavailable"` and the agent records a gap. The Verifier
  marks the citation as `lookup_failed` and **keeps** the study, so outages never silently remove evidence.
- An optional `NCBI_API_KEY` can be leveraged to raise the limit to 10 per second. However, it is not used here.
- Docs: <https://www.ncbi.nlm.nih.gov/books/NBK25497/>

### OpenAlex
Used by: literature researcher (`/works` title-and-abstract search).

- **Without a key:** a free daily budget of **$0.10 shared by every client on the same IP address**. A search
  costs $0.001, so roughly **100 searches per day per IP**. Each response's `meta.notice` reports this.
- **How where you run the project from impacts this:**
  - *Home network:* usually fine (about 2 searches per literature run).
  - *Campus network:* many users may share one public IP and this can use up the budget earlier in the day.
  - *Cloud Run:* outgoing requests use IP addresses shared with other Google Cloud customers, so the
    budget may already be spent. So we can expect OpenAlex to be more unreliable there without a key.
  - *Evals:* a full run of 20 questions uses a large share of the daily budget.
- **At the limit:** the search returns `"unavailable"`. The agent records a gap and continues with PubMed and
  PsyArXiv; it never crashes. What is lost is mostly non-medical journals (urban planning, social science).
- **Future Solution:** if we were to expand this project, we would obtain a free key from <https://openalex.org/settings/api> which then gives us about $1/day (about 1,000
  searches). We would then set an `OPENALEX_API_KEY` in `.env`; the remaining code would use this automatically, and it also works without it.

### OSF (PsyArXiv preprints)
Used by: literature researcher (`/v2/preprints/`).

- **Without a key:** no limit observed at our volume of requests.
- **Search limitation:** OSF matches based on **titles only, one term per request**. The tool searches the first
  concept's synonyms in titles (up to 5 requests) and keeps preprints whose title or abstract mentions the
  other concepts. A term absent from titles finds nothing; for example, no PsyArXiv title contains "walkability".
- Each version of a preprint is listed separately; the tool keeps the highest version.
- Preprints are not peer reviewed. The rubric subtracts a point, and a caveat is added.

### Crossref
Used by: Verifier (`/works/{doi}`, which includes Retraction Watch data under `updated-by`).

- **Without a contact address:** concurrency is limited. Three simultaneous requests returned HTTP 429.
- **Our usage:** one request per cited DOI, **one at a time**, with retries on 429. For about 10 papers, this takes a
  few seconds.
- **If it still fails:** the citation is marked `lookup_failed`, and PubMed is used alone when the paper has
  a PMID.
- Crossref offers a faster "polite pool" for clients that send a contact email (`mailto=`). It is
  deliberately not used, so no personal contact details are sent.
- Docs: <https://www.crossref.org/documentation/retrieve-metadata/rest-api/>

### Census Data API
Used by: none of the agents. Only the optional `uv run python -m scripts.build_data --check-census` command (`/data/2023/acs/acs5`).

- **Key:** `CENSUS_API_KEY` in `.env` is required for this check, and the command exits even if it is not set.
- **Our usage:** one request for a single sample tract. The check compares it with the prepared study data to confirm that the bulk download data matches the official API response. See `docs/study/census-api-comparison.json`.
- **Why it is optional:** the dataset build downloads the ACS Summary Files in bulk from `www2.census.gov`, which does not need a key and contains the same estimates and margins of error. The agents never call the Census API when answering questions.
- **If it fails:** the command reports that the check failed. Nothing else is affected, and the key value is never logged.
- Setup steps are in `docs/study/README.md` under "Census access".
- Docs: <https://www.census.gov/data/developers/guidance/api-user-guide.html>

## Settings in code

| Setting | Where | Value |
|---|---|---|
| Requests at a time per service | `apis.py` `CONCURRENCY` | Crossref 1, PubMed 2, OpenAlex 2, OSF 2 |
| Retries on HTTP 429 | `apis.py` `RETRIES` | 3, honoring `Retry-After` up to 10s |
| Request timeout | `apis.py` `TIMEOUT` | 20s |
| Papers per search | `literature.py` `PER_SEARCH` | 8 per database |
| PsyArXiv title requests per search | `literature.py` `OSF_TERMS` | 5 |
| Searches per run | `app.py` `create_literature_agent` | 2 per database; 10 `record_study`; 25 tool calls total |

## Tests never use the network

Tests replay from real responses saved in `tests/verify_responses.json` and `tests/literature_responses.json`.
They are refreshed only when an API changes its response format or additional test cases are added:

```sh
uv run python -m scripts.record_verify_responses
uv run python -m scripts.record_literature_responses
```

Both of the above scripts use the network, need no keys, and don't save if a request fails.

## Data build sources

Only needed when rebuilding the dataset. Public downloads are cached under `data/raw/study`.

| Script | Hosts | Key |
|---|---|---|
| `scripts/build_data.py` | Census (www2.census.gov, tigerweb), CDC PLACES (chronicdata.cdc.gov), EPA (edg.epa.gov), Harvard Dataverse (crosswalk) | None. `--check-census` needs `CENSUS_API_KEY` |
| `scripts/build_environment.py` | ParkServe / TPL, OpenStreetMap Overpass (overpass-api.de) | None. Overpass is a shared public server with its own fair-use limits |
