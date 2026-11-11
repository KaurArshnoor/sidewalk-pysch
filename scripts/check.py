"""Exercise all three agents through the live /chat route on the real study snapshot."""
import argparse
import asyncio
import json
import os

import httpx

from data import ROOT


async def main():
    from app import app

    os.environ["SIDEWALK_DATASET"] = str(ROOT / "data/processed/study")
    questions = {
        "built_environment": "Summarize walkability, park access, and mapped gathering-place amenities for the full loaded 20-metro study. State units, available tract counts, how park access was derived, and mapping/coverage limits.",
        "health_demographics": "Summarize frequent mental distress, poverty, and household income in the full loaded 20-metro study. Distinguish modeled tract estimates from direct observations.",
        "analyst": "Compare unadjusted and demographically adjusted walkability–frequent-mental-distress associations in the full study. Also fit an adjusted park_access_pct interaction. Return coefficients, intervals, sample sizes, exclusions and causal/spatial caveats.",
    }
    results = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
        metadata = (await client.get("/metadata")).json()
        assert metadata["synthetic"] is False and metadata["scope"]["top20_verified"]
        for role, question in questions.items():
            print(f"Checking {role} on real study data", flush=True)
            response = await client.post("/chat", json={"message": question, "agent": role})
            response.raise_for_status()
            result = response.json()
            assert result["mode"] == "live" and result["result"]["status"] == "ok"
            assert result["result"]["findings"] and all(c["agent"] == role for c in result["tool_calls"])
            if role == "built_environment":
                columns = {c for f in result["result"]["findings"] for c in f["evidence"].get("statistics", {})}
                assert {"walkability", "park_access_pct", "amenities_per_1000"} <= columns
                assert any(s["id"] == "osm-gathering-places" for s in result["result"]["sources"])
            if role == "analyst":
                models = [f for f in result["result"]["findings"] if "estimate" in f["evidence"]]
                assert {f["evidence"]["adjusted"] for f in models} == {True, False}
                assert any(f["evidence"].get("moderator") == "park_access_pct" for f in models)
                assert all(f["evidence"]["n_metros"] == 20 for f in models)
            results.append(result)
            output = ROOT / "docs/study/live-agent-checks.json"
            output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            print(f"{role}: HTTP {response.status_code}; {len(result['tool_calls'])} tools", flush=True)
        print("Checking selected-metro analysis", flush=True)
        scoped = await client.post("/chat", json={"agent": "analyst", "metro": "12060",
            "message": "Within this metro, compare unadjusted and adjusted walkability–mental-distress associations. What can these data establish about loneliness, social connection, and why?"})
        scoped.raise_for_status()
        body = scoped.json()
        models = [f["evidence"] for f in body["result"]["findings"] if "estimate" in f["evidence"]]
        assert {m["adjusted"] for m in models} == {True, False}
        assert all(m["metro_filter"] == "12060" and m["n_metros"] == 1 and m["ci95"] is None for m in models)
        results.append(body)
        (ROOT / "docs/study/live-agent-checks.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("All three agents and selected-metro LLM checks passed", flush=True)


async def compare_baselines():
    import os
    from datetime import datetime, timezone
    input_path = ROOT / "docs/study/live-agent-checks.json"
    output_path = ROOT / "docs/study/sample-comparisons.json"
    import litellm
    litellm.suppress_debug_info = True
    checks = json.loads(input_path.read_text())
    rows = []
    for check in checks:
        question = check['result']['question']
        response = await litellm.acompletion(model=os.environ['SIDEWALK_MODEL'],
            messages=[{'role': 'user', 'content': question}],
            vertex_location=os.getenv('VERTEX_LOCATION', 'global'), timeout=60, num_retries=0)
        role = check.get('agent', check['result']['agent'])
        rows.append({'agent': role, 'question': question, 'model': os.environ['SIDEWALK_MODEL'],
                     'checked_at': datetime.now(timezone.utc).isoformat(),
                     'baseline_no_system_no_tools': response.choices[0].message.content,
                     'agent_result': check['result'], 'agent_tool_calls': check['tool_calls']})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        print(role + ': baseline saved', flush=True)


GRADER_QUERIES = (
    ("built_environment", "Across the 20 metros, what are the mean EPA walkability score and the number of tracts with a score? Give the scale."),
    ("health_demographics", "Is CDC PLACES a direct survey of each tract? What period do the ACS measures cover? Answer in two sentences."),
    ("study", "Across the 20 metros, compare unadjusted and adjusted walkability–mental-distress associations. How do they compare with published research on walkability and depression? Keep the report brief."),
)


async def grader_samples():
    from datetime import datetime, timezone

    import litellm

    from app import app

    litellm.suppress_debug_info = True
    model = os.environ["SIDEWALK_MODEL"]
    rows = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
        for role, question in GRADER_QUERIES:
            print(f"Recording {role} sample", flush=True)
            response = await client.post("/chat", json={"agent": role, "message": question})
            response.raise_for_status()
            body = response.json()
            if role == "study":
                assert body["result"]["report"] and len({c["agent"] for c in body["tool_calls"]}) >= 2
            else:
                assert body["result"]["findings"] and body["tool_calls"]
            baseline = await litellm.acompletion(
                model=model, messages=[{"role": "user", "content": question}],
                vertex_location=os.getenv("VERTEX_LOCATION", "global"), timeout=60, num_retries=0,
            )
            rows.append({"agent": role, "question": question, "model": model,
                         "checked_at": datetime.now(timezone.utc).isoformat(),
                         "baseline_no_system_no_tools": baseline.choices[0].message.content,
                         "agent_response": body})
    path = ROOT / "evals/results/grader-samples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} grader comparisons to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the real LLM agents and optionally capture no-tools baselines")
    parser.add_argument("--baselines", action="store_true", help="Also make three same-model calls without tools or a system prompt")
    parser.add_argument("--grader-samples", action="store_true", help="Record the three README examples and no-tools baselines")
    args = parser.parse_args()
    if args.grader_samples:
        asyncio.run(grader_samples())
    else:
        asyncio.run(main())
        if args.baselines:
            asyncio.run(compare_baselines())
