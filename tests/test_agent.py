"""The agent loop validates every tool call before anything runs (scripted model, no network)."""
from __future__ import annotations

import json

from headroom.agent.operator import run
from headroom.agent.tools import SimulateExpansion, dispatch, tool_schemas


class FakeSite:
    def __init__(self):
        self.calls = []

    def site_status(self):
        return {"site_kw": 812.0}

    def simulate_expansion(self, args: SimulateExpansion):
        self.calls.append(args.extra_ports)
        return {"extra_ports": args.extra_ports, "new_peak_risk": 0.004}


class ScriptedModel:
    """Stands in for the model in tests: replays a fixed list of responses."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, **kwargs):
        self.seen.append(messages[-1])
        return {"choices": [{"message": self.replies.pop(0)}]}


def _call(name, arguments, call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_schemas_are_strict():
    for tool in tool_schemas():
        params = tool["function"]["parameters"]
        assert params.get("additionalProperties") is False
        assert set(params.get("required", [])) == set(params.get("properties", {}))


def test_invalid_call_is_returned_not_run():
    site = FakeSite()
    result, ok = dispatch(site, "simulate_expansion", json.dumps({"extra_ports": -5}))
    assert not ok and result["error"] == "invalid arguments" and site.calls == []
    result, ok = dispatch(site, "simulate_expansion", json.dumps({"extra_ports": 8, "force": True}))
    assert not ok and site.calls == []
    result, ok = dispatch(site, "delete_meter", "{}")
    assert not ok


def test_loop_recovers_from_a_bad_call():
    site = FakeSite()
    model = ScriptedModel([
        {"content": "Checking expansion.", "tool_calls": [_call("simulate_expansion", {"extra_ports": "twelve"})]},
        {"content": "", "tool_calls": [_call("simulate_expansion", {"extra_ports": 12}, "c2")]},
        {"content": "Adding 12 chargers carries a 0.4% risk of a new peak."},
    ])
    turn = run(site, "Can I add 12 chargers?", client=model)
    assert site.calls == [12]
    assert [s.ok for s in turn.steps if s.kind == "tool"] == [False, True]
    assert "0.4%" in turn.answer
