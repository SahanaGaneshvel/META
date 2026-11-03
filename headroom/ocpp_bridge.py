"""Headroom's decisions as OCPP 1.6J commands: the open protocol commercial chargers speak.

Every 15 minutes the site sends each charger a SetChargingProfile with the power the planner chose
and the safety layer allowed for the car on it (0 W for an idle charger, so a car that plugs in waits
for the next decision instead of adding unplanned load). The profile is absolute, in watts, valid for
that interval only: if the site controller goes quiet, the profile expires and the charger falls back
to its own default limit, which the installer sets to the static safe cap.

`CentralSystem` is the site side; `SimulatedCharger` is a charger that accepts profiles and records
them, for tests and the demo (scripts/ocpp_demo.py). Messages are built and schema-validated by the
open-source `ocpp` library (Mobility House, MIT), used unmodified.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import websockets
from ocpp.routing import on
from ocpp.v16 import ChargePoint, call, call_result
from ocpp.v16.enums import Action, ChargingProfileStatus, RegistrationStatus

SUBPROTOCOL = "ocpp1.6"
INTERVAL_S = 900


def charging_profile(profile_id: int, kw: float, start: datetime, seconds: int = INTERVAL_S) -> dict:
    """One interval's absolute limit in watts (OCPP 1.6 ChargingProfile, TxDefaultProfile)."""
    end = start + timedelta(seconds=seconds)
    return {"charging_profile_id": profile_id, "stack_level": 1,
            "charging_profile_purpose": "TxDefaultProfile", "charging_profile_kind": "Absolute",
            "valid_from": start.isoformat(), "valid_to": end.isoformat(),
            "charging_schedule": {"duration": seconds, "start_schedule": start.isoformat(),
                                  "charging_rate_unit": "W",
                                  "charging_schedule_period": [{"start_period": 0, "limit": round(kw * 1000, 1)}]}}


class SiteLink(ChargePoint):
    """The central system's connection to one charger."""

    @on(Action.boot_notification)
    def on_boot(self, charge_point_vendor: str, charge_point_model: str, **_):
        return call_result.BootNotification(current_time=datetime.now(timezone.utc).isoformat(), interval=300,
                                            status=RegistrationStatus.accepted)


@dataclass
class CentralSystem:
    """Accepts charger connections and pushes one limit per charger per interval."""
    links: dict[str, SiteLink] = field(default_factory=dict)
    sent: list[dict] = field(default_factory=list)

    async def _handler(self, websocket) -> None:
        charger_id = websocket.request.path.strip("/")
        link = SiteLink(charger_id, websocket)
        self.links[charger_id] = link
        try:
            await link.start()
        except websockets.ConnectionClosed:
            self.links.pop(charger_id, None)

    def serve(self, host: str = "127.0.0.1", port: int = 9000):
        return websockets.serve(self._handler, host, port, subprotocols=[SUBPROTOCOL])

    async def send_limits(self, limits_kw: dict[str, float], start: datetime, profile_id: int) -> dict[str, str]:
        """Send every charger its limit for the interval starting at `start`; returns each one's status."""
        async def one(charger_id: str, kw: float) -> tuple[str, str]:
            request = call.SetChargingProfile(connector_id=1,
                                              cs_charging_profiles=charging_profile(profile_id, kw, start))
            response = await self.links[charger_id].call(request)
            self.sent.append({"charger": charger_id, "start": start.isoformat(), "kw": kw,
                              "status": str(response.status)})
            return charger_id, str(response.status)

        return dict(await asyncio.gather(*(one(c, kw) for c, kw in limits_kw.items())))


class SimulatedCharger(ChargePoint):
    """A charger that accepts power limits and records them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limits_w: list[tuple[str, float]] = []

    @on(Action.set_charging_profile)
    def on_profile(self, connector_id: int, cs_charging_profiles: dict, **_):
        schedule = cs_charging_profiles["charging_schedule"]
        assert schedule["charging_rate_unit"] == "W"
        self.limits_w.append((cs_charging_profiles["valid_from"], schedule["charging_schedule_period"][0]["limit"]))
        return call_result.SetChargingProfile(status=ChargingProfileStatus.accepted)

    async def boot(self) -> None:
        await self.call(call.BootNotification(charge_point_vendor="Headroom", charge_point_model="simulated"))


async def connect_charger(charger_id: str, url: str) -> tuple[SimulatedCharger, asyncio.Task]:
    websocket = await websockets.connect(f"{url}/{charger_id}", subprotocols=[SUBPROTOCOL])
    charger = SimulatedCharger(charger_id, websocket)
    task = asyncio.create_task(charger.start())
    await charger.boot()
    return charger, task
