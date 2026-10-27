"""Check every Meta Model API feature Headroom depends on. Needs MODEL_API_KEY in .env.

    python scripts/api_smoke.py                 # all checks
    python scripts/api_smoke.py tools search    # a subset

Checks are independent, so a failure points at exactly one feature.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headroom.config import DATA  # noqa: E402
from headroom.meta.client import SAM_MODEL, as_data_url, make_client, segment, spark_model  # noqa: E402
from headroom.meta.sam_output import parse_boxes  # noqa: E402

TARIFF_PDF = DATA / "docs" / "e-4-tou_effective_2026-07-01.pdf"
STILL = DATA / "video" / "lot_close.jpg"
CLIP = DATA / "video" / "lot_close_5fps.mp4"


def check_models(c):
    ids = {m.id for m in c.models.list().data}
    print("   available:", ", ".join(sorted(ids)))
    missing = {spark_model(), SAM_MODEL} - ids
    assert not missing, f"not available to this key: {missing}"


def check_tools(c):
    tool = {"type": "function", "function": {
        "name": "site_headroom",
        "description": "P90 spare capacity in kW at the site for one 15-minute slot.",
        "parameters": {"type": "object", "properties": {"slot": {"type": "string", "description": "ISO 8601 local"}},
                       "required": ["slot"], "additionalProperties": False},
        "strict": True}}
    messages = [{"role": "user", "content": "What is our charging headroom at 2017-06-21T18:00? Use the tool."}]
    first = c.chat.completions.create(model=spark_model(), messages=messages, tools=[tool], reasoning_effort="low")
    reply = first.choices[0].message
    assert reply.tool_calls, f"no tool call (finish_reason={first.choices[0].finish_reason})"
    call = reply.tool_calls[0]
    print("   called:", call.function.name, call.function.arguments)
    # The full assistant message must precede the tool result, or the API returns 400.
    messages += [reply.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"p90_headroom_kw": 142.5})}]
    final = c.chat.completions.create(model=spark_model(), messages=messages, tools=[tool], reasoning_effort="low")
    text = final.choices[0].message.content or ""
    print("   answer:", text[:160])
    assert "142" in text, "the tool result was not used"


def check_search(c):
    r = c.responses.create(
        model=spark_model(),
        input="Is there a current or forecast CAISO Flex Alert or heat advisory for the San Francisco Bay Area "
              "this week? Give dates and hours.",
        tools=[{"type": "web_search", "search_context_size": "low",
                "user_location": {"country": "US", "city": "Palo Alto", "timezone": "America/Los_Angeles"}}],
    )
    cites = [a.url for item in r.output if item.type == "message" for part in item.content
             if part.type == "output_text" for a in (part.annotations or []) if a.type == "url_citation"]
    print("   answer:", r.output_text[:200].replace("\n", " "))
    print(f"   {len(cites)} citations:", cites[:3])
    assert cites, "no citations returned"


def check_pdf(c):
    schema = {"type": "object", "additionalProperties": False,
              "required": ["schedule", "peak_demand_per_kw_summer", "max_demand_per_kw_summer", "demand_interval_minutes"],
              "properties": {"schedule": {"type": "string"},
                             "peak_demand_per_kw_summer": {"type": "number"},
                             "max_demand_per_kw_summer": {"type": "number"},
                             "demand_interval_minutes": {"type": "integer"}}}
    r = c.responses.create(
        model=spark_model(),
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Extract these fields from the tariff schedule."},
            {"type": "input_file", "filename": TARIFF_PDF.name, "file_data": as_data_url(TARIFF_PDF)}]}],
        text={"format": {"type": "json_schema", "name": "tariff_fields", "schema": schema, "strict": True}},
        reasoning={"effort": "low"},
    )
    got = json.loads(r.output_text)
    print("   extracted:", got)
    # Values printed in the PDF itself (Section C and the Billing Demand note).
    assert abs(got["peak_demand_per_kw_summer"] - 30.18) < 0.005
    assert abs(got["max_demand_per_kw_summer"] - 21.27) < 0.005
    assert got["demand_interval_minutes"] == 15


def check_sam_image(c):
    text = segment(c, {"type": "input_image", "image_url": as_data_url(STILL)}, "car")
    cars = parse_boxes(text).get(0, [])
    print(f"   cars in still: {len(cars)}; raw start: {text[:120]!r}")
    assert cars, "no cars found (compare the raw output with the parser)"


def check_sam_video(c):
    t = time.time()
    frames = parse_boxes(segment(c, {"type": "input_video", "video_url": as_data_url(CLIP)}, "car"))
    tracks = {b.object_id for boxes in frames.values() for b in boxes}
    print(f"   frames with cars: {len(frames)}, distinct tracks: {len(tracks)}, {time.time() - t:.0f}s")
    assert frames, "no frames returned"


CHECKS = {"models": check_models, "tools": check_tools, "search": check_search, "pdf": check_pdf,
          "sam_image": check_sam_image, "sam_video": check_sam_video}


def main(selected: list[str]):
    client = make_client()
    results = {}
    for name in selected or CHECKS:
        print(f"-- {name}")
        try:
            CHECKS[name](client)
            results[name] = "PASS"
        except Exception as exc:  # keep going: each check is independent
            results[name] = f"FAIL {type(exc).__name__}: {exc}"
            traceback.print_exc(limit=1)
    print("\nSUMMARY")
    for name, outcome in results.items():
        print(f"  {name:10s} {outcome[:150]}")
    return all(v == "PASS" for v in results.values())


if __name__ == "__main__":
    unknown = set(sys.argv[1:]) - set(CHECKS)
    if unknown:
        sys.exit(f"unknown checks {unknown}; choose from {list(CHECKS)}")
    sys.exit(0 if main(sys.argv[1:]) else 1)
