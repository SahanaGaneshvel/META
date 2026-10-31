"""The operator agent loop: Muse Spark (or Muse Glimmer on site when the internet is down) calling validated tools.

Returns a trace of every step (model text, tool call, validated result) for the control room to
show, so the judge sees the model propose and the math decide.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..meta.client import ModelClient
from .tools import SiteContext, dispatch, tool_schemas

SYSTEM = """You are the operator of an EV-charging site that shares one electricity meter with a building.
Your job: keep drivers charged without setting a new monthly demand peak, and answer the owner's
questions with numbers from your tools, never guesses. The controller and its safety layer enforce
every limit; you propose, they decide. Before acting, say in one short sentence what you will check.
Write for a site manager in plain sentences; never show tool field names. Use each number exactly as
a tool returned it (rounding is fine) and say what it is, e.g. "the forecast's P99 is 776 kW"; don't
work out new numbers yourself. Write money as $1,449, shares as percentages (0.998 is 99.8%), and
times as 5:30 pm. The comparison controller is "standard smart charging".
On a grid alert, size any forecast adjustment from the evidence the alert tool returns about this
building on past alert days, cite it as the source, and report what the re-plan changed.
When a tool rejects something, explain why to the owner in plain words. Keep answers to a few lines."""

DEMO_REQUESTS = [  # (site time, request): recorded by scripts/record_demo.py, in this order, on a fresh site
    ("2017-06-21T15:00", "What's the situation at the site right now?"),
    ("2017-06-21T15:00", "Is there any grid alert today, and should we adjust the plan for it?"),
    ("2017-06-21T18:00", "Fleet vans need 40 kWh each by 6 am tomorrow, six of them, plugging in at 7 pm. Can we do it?"),
    ("2017-06-21T18:00", "Why did we charge the way we did at 5:30 pm?"),
    ("2017-06-21T18:00", "Can I add 12 more chargers? What would it cost me?"),
    ("2017-06-21T18:00", "Would Headroom work at our other buildings too?"),
]

_NUMBER = re.compile(r"(?<![\w.:])\d[\d,]*(?:\.\d+)?")
_TIMES = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?|\d{1,2}:\d{2}"  # dates, clock times, "4-9pm", "6 am"
                    r"|\d{1,2}\s*(?:-|\u2013|to)\s*\d{1,2}\s*(?:am|pm|h)\b|\d{1,2}\s*(?:am|pm|h)\b"
                    r"|\d{1,2}[\s-]*min(?:ute)?s?\b|\d{1,2}(?:st|nd|rd|th)\b", re.IGNORECASE)  # "15-minute", "95th"


def _numbers(text: str) -> list[tuple[str, float]]:
    text = _TIMES.sub(" ", text)
    return [(m.group(), float(m.group().replace(",", ""))) for m in _NUMBER.finditer(text)]


def _values(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _values(x)]
    if isinstance(obj, (list, tuple)):
        return [v for x in obj for v in _values(x)]
    if isinstance(obj, str):
        return [v for _, v in _numbers(obj)]
    return []


def unsupported_numbers(answer: str, results: list[dict], request: str = "") -> list[str]:
    """Numbers in the answer that match no tool result (as given, or as a percentage) and no number in
    the owner's request (or a product of two of them, e.g. 6 vans x 40 kWh). A number may round its
    source to the digits it shows: "41" for 40.7, "99.8" for 0.998 as a percentage, not "798" for 799."""
    asked = [v for _, v in _numbers(request)] if isinstance(request, str) else []
    known = _values(results) + asked + [a * b for i, a in enumerate(asked) for b in asked[i + 1:]]
    known += [v * 100 for v in known if abs(v) <= 1.5]

    def found(text: str, x: float) -> bool:
        decimals = len(text.split(".")[1]) if "." in text else 0
        return any(abs(x - v) <= 0.5 * 10 ** -decimals + 1e-9 for v in known)
    return [text for text, x in _numbers(answer) if not found(text, x)]


@dataclass
class Step:
    kind: str                # "say" | "tool"
    text: str = ""
    tool: str = ""
    arguments: str = ""
    result: dict = field(default_factory=dict)
    ok: bool = True


@dataclass
class Turn:
    steps: list[Step]
    answer: str
    messages: list[dict]
    answered_by: str = "spark"  # the role that served the last model call (FailoverClient)
    unsupported: list[str] = field(default_factory=list)  # numbers no tool result backs up


def run(context: SiteContext, request: str, history: list[dict] | None = None, client: ModelClient | None = None,
        max_rounds: int = 6, reasoning_effort: str = "low") -> Turn:
    client = client or ModelClient("spark")
    messages = list(history or [{"role": "system", "content": SYSTEM}])
    messages.append({"role": "user", "content": request})
    steps: list[Step] = []
    for _ in range(max_rounds):
        response = client.chat(messages, tools=tool_schemas(), tool_choice="auto", reasoning_effort=reasoning_effort)
        message = response["choices"][0]["message"]
        if message.get("content"):
            steps.append(Step(kind="say", text=message["content"]))
        calls = message.get("tool_calls") or []
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if not calls:
            answer = message.get("content") or ""
            results = [st.result for st in steps if st.kind == "tool" and st.ok]
            return Turn(steps=steps, answer=answer, messages=messages,
                        answered_by=getattr(client, "answered_by", "spark"),
                        unsupported=unsupported_numbers(answer, results, request))
        for call in calls:
            fn = call["function"]
            result, ok = dispatch(context, fn["name"], fn.get("arguments", "{}"))
            steps.append(Step(kind="tool", tool=fn["name"], arguments=fn.get("arguments", ""), result=result, ok=ok))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, default=str)})
    return Turn(steps=steps, answer="(stopped after the maximum number of tool rounds)", messages=messages)
