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


def test_spend_guard_books_the_worst_case_first(tmp_path, monkeypatch):
    import pytest

    from headroom.meta import client as mc

    monkeypatch.setattr(mc, "SPEND_LOG", tmp_path / "spend.json")
    monkeypatch.setenv("HEADROOM_BUDGET_USD", "10")
    c = mc.ModelClient("spark", mode="live", recordings=tmp_path)
    c.config = mc.RoleConfig("spark", "https://api.meta.ai/v1", "test-key", "muse-spark-1.3")
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}

    class Reply:
        def model_dump(self, **kw):
            return {"choices": [{"message": {"content": "ok"}}], "usage": usage}

    seen = []

    class Api:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    seen.append(mc.spent_usd())  # the worst case is already booked when the call goes out
                    return Reply()

    c._client = Api
    c.chat([{"role": "user", "content": "hi"}])
    assert seen == [pytest.approx(mc.WORST_CALL_USD)]
    assert mc.spent_usd() == pytest.approx(1.25)  # then settled to the reported usage

    (tmp_path / "spend.json").write_text('{"usd": 6.0}')  # 6 + 4.50 could pass 10: refused, nothing sent
    with pytest.raises(mc.BudgetExceeded):
        c.chat([{"role": "user", "content": "again"}])
    assert len(seen) == 1 and mc.spent_usd() == 6.0

    monkeypatch.setenv("HEADROOM_BUDGET_USD", "100000")  # .env can't lift the hard cap
    (tmp_path / "spend.json").write_text('{"usd": 136.0}')
    with pytest.raises(mc.BudgetExceeded):
        c.chat([{"role": "user", "content": "once more"}])

    usage.clear()  # no usage reported: the worst case stays booked
    (tmp_path / "spend.json").write_text('{"usd": 0}')
    c.chat([{"role": "user", "content": "no usage"}])
    assert mc.spent_usd() == pytest.approx(mc.WORST_CALL_USD)


def test_failover_to_glimmer_when_the_link_is_down(tmp_path):
    import pytest

    from headroom.meta import client as mc

    class Down:
        def chat(self, messages, **kwargs):
            raise mc.LinkDown("cloud unreachable")

    fc = mc.FailoverClient(recordings=tmp_path)
    fc.backup = ScriptedModel([{"content": "Glimmer here: site at 812 kW."}])
    fc.backup.config = mc.RoleConfig("glimmer", "", "", "muse-glimmer")
    fc.primary = Down()
    turn = run(FakeSite(), "Status?", client=fc)
    assert turn.answered_by == "glimmer" and "812" in turn.answer

    fc.backup = Down()
    with pytest.raises(mc.LinkDown):  # neither model: the caller is told, charging is unaffected
        fc.chat([{"role": "user", "content": "hi"}])
