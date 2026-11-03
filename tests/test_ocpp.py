"""Headroom's limits reach chargers over OCPP 1.6J unchanged (local websockets, simulated chargers)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from headroom.ocpp_bridge import CentralSystem, connect_charger


def test_limits_arrive_at_each_charger_as_sent():
    async def scenario():
        site = CentralSystem()
        async with site.serve(port=9123):
            chargers = dict([await connect_charger(c, "ws://127.0.0.1:9123") for c in ("HR-01", "HR-02")])
            plan = [{"HR-01": 7.2, "HR-02": 0.0}, {"HR-01": 3.35, "HR-02": 6.6}]
            start = datetime(2017, 6, 21, 17, 0, tzinfo=timezone.utc)
            statuses = [await site.send_limits(limits, start.replace(minute=15 * i), profile_id=i + 1)
                        for i, limits in enumerate(plan)]
            for task in chargers.values():
                task.cancel()
            return chargers, statuses

    chargers, statuses = asyncio.run(scenario())
    assert all(s == "Accepted" for step in statuses for s in step.values())
    received = {c.id: [w for _, w in c.limits_w] for c in chargers}
    assert received == {"HR-01": [7200.0, 3350.0], "HR-02": [0.0, 6600.0]}
