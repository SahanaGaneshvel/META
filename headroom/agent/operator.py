"""The operator agent loop: Muse Spark Open (or Glimmer offline) calling validated tools.

Returns a trace of every step (model text, tool call, validated result) for the control room to
show, so the judge sees the model propose and the math decide.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..meta.client import ModelClient
from .tools import SiteContext, dispatch, tool_schemas

SYSTEM = """You are the operator of an EV-charging site that shares one electricity meter with a building.
Your job: keep drivers charged without setting a new monthly demand peak, and answer the owner's
questions with numbers from your tools, never guesses. The controller and its safety layer enforce
every limit; you propose, they decide. Before acting, say in one short sentence what you will check.
When a tool rejects something, explain why to the owner in plain words. Keep answers short."""


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
            return Turn(steps=steps, answer=message.get("content") or "", messages=messages)
        for call in calls:
            fn = call["function"]
            result, ok = dispatch(context, fn["name"], fn.get("arguments", "{}"))
            steps.append(Step(kind="tool", tool=fn["name"], arguments=fn.get("arguments", ""), result=result, ok=ok))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, default=str)})
    return Turn(steps=steps, answer="(stopped after the maximum number of tool rounds)", messages=messages)
