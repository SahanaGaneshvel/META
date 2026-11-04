"""Data for the web demo (docs/demo/data.json): the real heat-wave day step by step, the recorded
operator conversations, and the results. Everything is read from the simulation, the recordings and
data/results/; nothing is typed in.

    python scripts/export_demo.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["HEADROOM_MODEL_MODE"] = "replay"  # recorded model answers only: free, no key
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from headroom.agent.operator import DEMO_REQUESTS, SYSTEM, VOICE_REQUEST_AT, run  # noqa: E402
from headroom.agent.site import TARIFF_DOCS, ReplaySite  # noqa: E402
from headroom.capacity import hosting_capacity  # noqa: E402
from headroom.config import DATA, LOCAL_TZ, ROOT  # noqa: E402
from headroom.control.headroom_mpc import HeadroomMPC  # noqa: E402
from headroom.forecast.precomputed import PrecomputedForecaster  # noqa: E402
from headroom.meta.client import FailoverClient, ModelClient, transcribe  # noqa: E402
from headroom.sim.engine import simulate  # noqa: E402
from headroom.tariff import monthly_bills  # noqa: E402

DAY = "2017-06-21"
OUT = ROOT / "docs" / "demo" / "data.json"
R = DATA / "results"


def r1(x) -> float:
    return round(float(x), 1)


def day_steps(site: ReplaySite) -> tuple[list[dict], dict]:
    sc, spec = site.scenario, site.spec
    local = sc.index.tz_convert(LOCAL_TZ)
    steps = np.flatnonzero(np.asarray(local.strftime("%Y-%m-%d") == DAY))
    ports = sorted(set(sc.port.tolist()))
    per_port: dict[int, list[float]] = {}

    def observe(t, active, kw):
        if t in step_set:
            limits = [0.0] * len(ports)
            for car, p in zip(active, kw):
                limits[ports.index(sc.port[car])] += float(p)
            per_port[t] = limits

    step_set = set(steps.tolist())
    hr = simulate(sc, HeadroomMPC(PrecomputedForecaster(), spec, site.site["building_kw"]), observer=observe)
    assert np.allclose(hr.ev_kw, site.results["headroom"].ev_kw), "demo must match the app's simulation"
    llf = site.results["llf"]
    labels = site.labels
    window = labels.filter(like="demand:peak").any(axis=1).to_numpy()
    building = pd.Series(sc.building_kw, index=sc.index)

    def added_demand(ev: np.ndarray, t: int) -> float:
        b = building.iloc[: t + 1]
        alone = monthly_bills(b, spec)["demand_usd"]
        return float((monthly_bills(b + ev[: t + 1], spec)["demand_usd"] - alone).sum())

    rows = []
    for t in steps:
        site_hr = sc.building_kw + hr.ev_kw
        same = (local.month == local[t].month) & (np.arange(len(sc.index)) < t)
        paid_any = float(site_hr[same].max()) if same.any() else 0.0
        paid_window = float(site_hr[same & window].max()) if (same & window).any() else 0.0
        q = site.forecast[t]
        rows.append({
            "time": local[t].strftime("%H:%M"), "window": bool(window[t]), "building": r1(sc.building_kw[t]),
            "ev_headroom": r1(hr.ev_kw[t]), "ev_standard": r1(llf.ev_kw[t]),
            "cars": int(((sc.arrive <= t) & (sc.depart > t)).sum()),
            "paid_kw": r1(paid_window if window[t] else paid_any), "trim": r1(hr.safety_clipped_kw[t]),
            "p50": [r1(v) for v in q[:24, 1]], "p99": [r1(v) for v in q[:24, 3]],
            "chargers": [r1(v) for v in per_port.get(t, [0.0] * len(ports))],
            "bill_headroom": round(added_demand(hr.ev_kw, t)), "bill_standard": round(added_demand(llf.ev_kw, t)),
        })
    peak = next(s for s in spec.seasons if 6 in s.months)
    return rows, {"service_kw": float(sc.service_kw), "chargers": len(ports), "sessions": int(len(site.sessions)),
                  "demand_rates": {d.name: d.usd_per_kw for d in peak.demand}}


def conversation(link_up: bool) -> list[dict]:
    site = ReplaySite(month="2017-06", n_ports=24).fork()
    client = FailoverClient("spark", "glimmer", mode="replay", link_up=link_up)
    history = [{"role": "system", "content": SYSTEM}]
    turns = []
    for when, question in DEMO_REQUESTS:
        site.now_local = when
        turn = run(site, question, history=history, client=client)
        history = turn.messages
        turns.append({
            "time": when[-5:], "question": question, "answer": turn.answer,
            "model": "Muse Glimmer (on site)" if turn.answered_by == "glimmer" else "Muse Spark (cloud)",
            "unsupported": turn.unsupported,
            "tools": [{"name": s.tool, "ok": s.ok, "arguments": json.loads(s.arguments or "{}"), "result": s.result}
                      for s in turn.steps if s.kind == "tool"],
        })
    return turns


def voice() -> dict:
    """The recorded spoken request (scripts/record_voice.py): transcript, then Muse Spark's answer."""
    import shutil

    wav = DATA / "voice" / "question.wav"
    text = transcribe(wav.read_bytes(), mode="replay")
    site = ReplaySite(month="2017-06", n_ports=24).fork()
    site.now_local = VOICE_REQUEST_AT
    turn = run(site, text, history=[{"role": "system", "content": SYSTEM}], client=ModelClient("spark", mode="replay"))
    shutil.copy(wav, OUT.parent / "question.wav")
    return {"time": VOICE_REQUEST_AT[-5:], "question": text, "answer": turn.answer, "model": "Muse Spark (cloud)",
            "unsupported": turn.unsupported,
            "tools": [{"name": s.tool, "ok": s.ok, "arguments": json.loads(s.arguments or "{}"), "result": s.result}
                      for s in turn.steps if s.kind == "tool"]}


def results() -> dict:
    final = pd.read_csv(R / "final_summary.csv", index_col="controller")
    b19 = pd.read_csv(R / "final_summary_pge-b-19_2026.csv", index_col="controller")
    cap = pd.read_csv(R / "capacity_summary_test.csv")
    cap_b19 = pd.read_csv(R / "capacity_summary_test_pge-b-19_2026.csv")
    hosting, hosting_b19 = hosting_capacity(cap), hosting_capacity(cap_b19)
    p = pd.read_csv(R / "portfolio_summary.csv")
    p90 = pd.read_csv(R / "portfolio_summary_share90.csv")
    both, both90 = p[p["chargers_standard_smart_charging"] > 0], p90[p90["chargers_standard_smart_charging"] > 0]
    site = ReplaySite(month="2017-06", n_ports=24)
    tariffs = {}
    for doc in TARIFF_DOCS:
        v = site.read_tariff_document(doc)["verification"]
        tariffs[doc] = {"fields": v["fields_correct"], "bill_error": v["bill_reproduction_worst_month_error"]}
    curve = {name: [{"chargers": int(r.n_ports), "months_served": round(float(r.months_drivers_served), 3)}
                    for r in cap[cap["controller"] == name].itertuples()] for name in ("llf", "headroom")}
    return {
        "standard_usd": round(final.loc["llf", "added_demand_usd"]), "headroom_usd": round(final.loc["headroom", "added_demand_usd"]),
        "oracle_usd": round(final.loc["headroom_oracle", "added_demand_usd"]),
        "served": round(float(final.loc["headroom", "energy_served"]), 4),
        "violations": int(final.loc["headroom", "service_violations"]),
        "chargers_headroom": int(hosting["headroom"]), "chargers_standard": int(hosting["llf"]),
        "capacity_curve": curve,
        "b19": {"standard_usd": round(b19.loc["llf", "added_demand_usd"]), "headroom_usd": round(b19.loc["headroom", "added_demand_usd"]),
                "chargers_headroom": int(hosting_b19["headroom"]), "chargers_standard": int(hosting_b19["llf"])},
        "buildings": [{"standard": int(r.chargers_standard_smart_charging), "headroom": int(r.chargers_headroom),
                       "use": str(r.primaryspaceusage)} for r in p.itertuples()],
        "portfolio": {"n": len(p), "sites": int(p["site_id"].nunique()),
                      "more": round(float((both["extra_chargers"] > 0).mean()), 3),
                      "fewer": round(float((both["extra_chargers"] < 0).mean()), 3),
                      "median": round(float((both["chargers_headroom"] / both["chargers_standard_smart_charging"] - 1).median()), 3),
                      "median_tight": round(float((both90["chargers_headroom"] / both90["chargers_standard_smart_charging"] - 1).median()), 3),
                      "violations": int(p["violations_at_reported_sizes"].sum())},
        "tariffs": tariffs,
        "ocpp": json.loads((R / "ocpp_demo.json").read_text()),
    }


def main() -> None:
    site = ReplaySite(month="2017-06", n_ports=24)
    steps, meta = day_steps(site)
    data = {"day": DAY, "meta": meta, "steps": steps, "spark": conversation(True), "glimmer": conversation(False),
            "voice": voice(), "results": results()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), default=str), encoding="utf-8")
    flagged = [t["question"] for t in data["spark"] + data["glimmer"] if t["unsupported"]]
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB); {len(steps)} steps; answers with unbacked numbers: {flagged}")


if __name__ == "__main__":
    main()
