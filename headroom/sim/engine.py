"""15-minute site simulator: a controller proposes per-car power, the safety layer has the last word.

At the start of interval t the controller knows the building load up to t-1 (the meter), the cars
plugged in now with their remaining energy and departure, and nothing about future arrivals.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .scenario import STEP_H, Scenario


@dataclass
class StepView:
    """What a controller may look at when deciding interval t."""
    t: int
    active: np.ndarray            # session ids plugged in and still needing energy
    remaining_kwh: np.ndarray     # per active session
    depart: np.ndarray            # per active session (interval index, exclusive)
    max_kw: np.ndarray            # per active session
    building_measured: np.ndarray  # building kW for intervals < t
    ev_history: np.ndarray        # total EV kW for intervals < t


class Controller(Protocol):
    name: str

    def decide(self, view: StepView, scenario: Scenario) -> np.ndarray: ...


@dataclass
class SafetyLayer:
    """Deterministic last line of defence; nothing upstream can override it.

    Guarantee: if the building's load rises by at most `ramp_kw` from one interval to the next, the
    site never exceeds `service_kw`. The default covers the largest measured 15-minute rise in the
    training year (51.4 kW) with margin (the test year's largest is 53.4 kW); stress tests with
    added intra-hour noise must widen it. Whether the assumption held is reported for every run
    (`metrics.ramp_breaches`).
    """
    service_kw: float
    ramp_kw: float = 60.0
    extra_cap_kw: np.ndarray | None = None  # optional per-interval site cap (e.g. a utility request)

    def apply(self, proposal: np.ndarray, view: StepView, scenario: Scenario) -> tuple[np.ndarray, float]:
        p = np.nan_to_num(np.asarray(proposal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if p.shape != view.active.shape:
            p = np.zeros(view.active.shape)
        upper = np.minimum(view.max_kw, view.remaining_kwh / STEP_H)
        p = np.clip(p, 0.0, upper)
        last = view.building_measured[-1] if view.t > 0 else scenario.building_kw[0]
        site_cap = self.service_kw
        if self.extra_cap_kw is not None:
            site_cap = min(site_cap, self.extra_cap_kw[view.t])
        budget = max(0.0, site_cap - last - self.ramp_kw)
        total = p.sum()
        clipped = 0.0
        if total > budget:
            p = p * (budget / total)
            clipped = total - budget
        return p, clipped


@dataclass
class Result:
    controller: str
    ev_kw: np.ndarray
    delivered_kwh: np.ndarray
    safety_clipped_kw: np.ndarray
    seconds: float
    extra: dict = field(default_factory=dict)


def simulate(scenario: Scenario, controller: Controller, safety: SafetyLayer | None = None) -> Result:
    safety = safety or SafetyLayer(scenario.service_kw)
    n_steps = len(scenario.index)
    remaining = scenario.request_kwh.astype(float).copy()
    ev_kw = np.zeros(n_steps)
    clipped = np.zeros(n_steps)
    started = time.perf_counter()
    for t in range(n_steps):
        active = np.flatnonzero((scenario.arrive <= t) & (scenario.depart > t) & (remaining > 1e-6))
        if active.size == 0:
            continue
        view = StepView(t=t, active=active, remaining_kwh=remaining[active], depart=scenario.depart[active],
                        max_kw=scenario.max_kw[active], building_measured=scenario.building_kw[:t],
                        ev_history=ev_kw[:t])
        kw, clipped[t] = safety.apply(controller.decide(view, scenario), view, scenario)
        remaining[active] -= kw * STEP_H
        ev_kw[t] = kw.sum()
    return Result(controller=controller.name, ev_kw=ev_kw, delivered_kwh=scenario.request_kwh - remaining,
                  safety_clipped_kw=clipped, seconds=time.perf_counter() - started)
