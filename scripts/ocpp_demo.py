"""Drive 24 chargers over OCPP 1.6J with Headroom's real decisions for the heat-wave evening.

    python scripts/ocpp_demo.py        # -> data/results/ocpp_demo.json

Re-runs June 2017 at 24 chargers (the app's Live replay), records the per-car power the safety layer
allowed for 4-9 pm on 21 June, and sends it interval by interval to 24 simulated chargers over local
websockets. Checks that every charger received exactly its planned limit and that each interval's
total matches the simulation. The chargers are simulated; the protocol path is the real one.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ocpp.charge_point import snake_to_camel_case  # noqa: E402

from headroom.config import DATA, LOCAL_TZ, PROCESSED  # noqa: E402
from headroom.control.headroom_mpc import HeadroomMPC  # noqa: E402
from headroom.forecast.precomputed import PrecomputedForecaster  # noqa: E402
from headroom.ocpp_bridge import CentralSystem, charging_profile, connect_charger  # noqa: E402
from headroom.sim.engine import simulate  # noqa: E402
from headroom.sim.scenario import make_scenario  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402

DAY, START, END, PORTS = "2017-06-21", "16:00", "21:00", 24


def planned_limits() -> tuple[list[pd.Timestamp], list[dict[str, float]], np.ndarray]:
    site = pd.read_parquet(PROCESSED / "site_15min.parquet")
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    scenario = make_scenario(site, sessions, "2017-06-01", "2017-06-30", PORTS, seed=1)
    local = scenario.index.tz_convert(LOCAL_TZ)
    window = (local >= pd.Timestamp(f"{DAY} {START}", tz=LOCAL_TZ)) & (local < pd.Timestamp(f"{DAY} {END}", tz=LOCAL_TZ))
    steps = set(np.flatnonzero(window))
    ports = sorted(set(scenario.port.tolist()))
    names = {p: f"HR-{i + 1:02d}" for i, p in enumerate(ports)}
    plan: dict[int, dict[str, float]] = {}

    def observe(t, active, kw):
        if t in steps:
            limits = {name: 0.0 for name in names.values()}  # idle chargers get 0 W until the next decision
            for car, p in zip(active, kw):
                limits[names[scenario.port[car]]] += float(p)
            plan[t] = limits

    result = simulate(scenario, HeadroomMPC(PrecomputedForecaster(), load_gold("e-4-tou_2026"), site["building_kw"]),
                      observer=observe)
    order = sorted(steps)
    return ([scenario.index[t] for t in order], [plan.get(t, {n: 0.0 for n in names.values()}) for t in order],
            result.ev_kw[order])


async def send(times, plan) -> tuple[dict, list[float], dict]:
    site = CentralSystem()
    async with site.serve(port=9000):
        connected = [await connect_charger(name, "ws://127.0.0.1:9000") for name in plan[0]]
        chargers = {c.id: c for c, _ in connected}
        round_trips, statuses = [], []
        for i, (when, limits) in enumerate(zip(times, plan)):
            started = time.perf_counter()
            statuses.append(await site.send_limits(limits, when.to_pydatetime(), profile_id=i + 1))
            round_trips.append((time.perf_counter() - started) * 1000)
        for _, task in connected:
            task.cancel()
    received = {cid: [w for _, w in c.limits_w] for cid, c in chargers.items()}
    return received, round_trips, {"accepted": sum(s == "Accepted" for step in statuses for s in step.values()),
                                   "messages": sum(len(step) for step in statuses)}


def main() -> None:
    times, plan, sim_total_kw = planned_limits()
    received, round_trips, counts = asyncio.run(send(times, plan))
    sent_w = {cid: [round(step[cid] * 1000, 1) for step in plan] for cid in plan[0]}
    assert all(len(received[cid]) == len(plan) for cid in sent_w), "a charger missed an interval"
    mismatch = max(abs(a - float(b)) for cid in sent_w for a, b in zip(sent_w[cid], received[cid]))
    total_gap = float(np.max(np.abs(np.array([sum(step.values()) for step in plan]) - sim_total_kw)))
    example = snake_to_camel_case(charging_profile(1, next(v for v in plan[len(plan) // 2].values() if v > 0),
                                                   times[len(times) // 2].to_pydatetime()))
    out = {"day": DAY, "window_local": f"{START}-{END}", "chargers": len(plan[0]), "intervals": len(plan),
           **counts, "max_limit_mismatch_w": mismatch, "max_interval_total_gap_kw": round(total_gap, 6),
           "median_round_trip_ms_all_chargers": round(float(np.median(round_trips)), 1),
           "peak_planned_ev_kw": round(max(sum(s.values()) for s in plan), 1),
           "example_set_charging_profile": {"connectorId": 1, "csChargingProfiles": example}}
    (DATA / "results" / "ocpp_demo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "example_set_charging_profile"}, indent=1))


if __name__ == "__main__":
    main()
