"""The operator agent's tools: strict schemas in, validated calls out.

The model only *proposes*. Each tool call is parsed against its pydantic model; a malformed call is
returned to the model as an error it can fix, never executed. Tools that change the plan go
through the planner and the safety layer like everything else.
"""
from __future__ import annotations

import json
from typing import Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Args):
    pass


class ExplainPlan(_Args):
    local_time: str = Field(description="ISO local time to explain, e.g. 2017-06-21T17:30")


class SetRequirement(_Args):
    label: str = Field(description="who the requirement is for, e.g. 'fleet vans'")
    vehicles: int = Field(ge=1, le=60)
    kwh_per_vehicle: float = Field(gt=0, le=150, description="energy each vehicle needs")
    plug_in_local: str = Field(description="ISO local time the vehicles plug in")
    ready_by_local: str = Field(description="ISO local deadline")


class SimulateExpansion(_Args):
    extra_ports: int = Field(ge=1, le=120, description="chargers to add on top of today's")


class GridEventAdjustment(_Args):
    start_local: str
    end_local: str
    building_multiplier: float = Field(ge=0.8, le=1.3, description="expected building load vs. normal, e.g. 1.08")
    source: str = Field(description="where the evidence comes from (alert id, URL, or document)")
    reason: str


class ReadTariff(_Args):
    document: Literal["cpau-e-4-tou-2026", "cpau-e-7-tou-2026"]


class GridAlerts(_Args):
    local_date: str = Field(description="ISO local date to check")


class SiteContext(Protocol):
    def site_status(self) -> dict: ...
    def explain_plan(self, args: ExplainPlan) -> dict: ...
    def set_requirement(self, args: SetRequirement) -> dict: ...
    def simulate_expansion(self, args: SimulateExpansion) -> dict: ...
    def grid_event_adjustment(self, args: GridEventAdjustment) -> dict: ...
    def read_tariff(self, args: ReadTariff) -> dict: ...
    def grid_alerts(self, args: GridAlerts) -> dict: ...


TOOLS: dict[str, tuple[type[_Args], str, str]] = {
    "get_site_status": (NoArgs, "site_status",
                        "Current load, plugged-in cars, demand ceilings already paid this month, and the next hours' headroom."),
    "explain_plan": (ExplainPlan, "explain_plan",
                     "Why the controller charged as it did at a given time: forecast band, ceilings, binding limits."),
    "set_requirement": (SetRequirement, "set_requirement",
                        "Ask whether a group of vehicles can get a given energy by a deadline. Returns the "
                        "probability it is met under the forecast, and adds it to the plan only if accepted."),
    "simulate_expansion": (SimulateExpansion, "simulate_expansion",
                           "Monte Carlo on real sessions: risk of a new monthly demand peak, energy served and "
                           "bill impact if this many chargers are added."),
    "propose_grid_event_adjustment": (GridEventAdjustment, "grid_event_adjustment",
                                      "Propose a temporary change to the building-load forecast for a grid event. "
                                      "Accepted only if it stays inside the forecast's calibrated band."),
    "read_tariff": (ReadTariff, "read_tariff",
                    "Extract a utility tariff into structured rates and verify it by reproducing real bills."),
    "check_grid_alerts": (GridAlerts, "grid_alerts",
                          "Grid-stress alerts (CAISO Flex Alerts) for a date, with their source."),
}


def tool_schemas() -> list[dict]:
    out = []
    for name, (model, _, description) in TOOLS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        out.append({"type": "function", "function": {"name": name, "description": description,
                                                      "parameters": schema, "strict": True}})
    return out


def dispatch(context: SiteContext, name: str, raw_arguments: str) -> tuple[dict, bool]:
    """Run one tool call. Returns (result, ok); errors go back to the model, never raise."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}", "available": list(TOOLS)}, False
    model, method, _ = TOOLS[name]
    try:
        args = model.model_validate_json(raw_arguments or "{}")
    except ValidationError as exc:
        return {"error": "invalid arguments", "details": json.loads(exc.json(include_url=False))}, False
    handler: Callable = getattr(context, method)
    try:
        return (handler() if model is NoArgs else handler(args)), True
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
        return {"error": f"{type(exc).__name__}: {exc}"}, False
