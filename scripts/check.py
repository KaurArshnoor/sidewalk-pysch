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
        "built_environment": "Summarize EPA walkability for the full loaded 20-metro study. State its scale, coverage, and interpolation limits. Identify missing park-access data.",
        "health_demographics": "Summarize frequent mental distress, poverty, and household income in the full loaded 20-metro study. Distinguish modeled tract estimates from direct observations.",
        "analyst": "Compare unadjusted and demographically adjusted walkability–frequent-mental-distress associations in the full study. Return coefficients, intervals, sample sizes, exclusions and causal/spatial caveats.",
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
            if role == "analyst":
                models = [f for f in result["result"]["findings"] if "estimate" in f["evidence"]]
                assert {f["evidence"]["adjusted"] for f in models} == {True, False}
                assert all(f["evidence"]["n_metros"] == 20 for f in models)
            results.append(result)
            output = ROOT / "docs/study/live-agent-checks.json"
            output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            print(f"{role}: HTTP {response.status_code}; {len(result['tool_calls'])} tools", flush=True)
    print("All three real-study LLM checks passed", flush=True)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the real LLM agents and optionally capture no-tools baselines")
    parser.add_argument("--baselines", action="store_true", help="Also make three same-model calls without tools or a system prompt")
    args = parser.parse_args()
    asyncio.run(main())
    if args.baselines:
        asyncio.run(compare_baselines())
