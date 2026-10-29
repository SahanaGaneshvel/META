"""Reference controllers. All EV-only: they see a fixed EV capacity, never the building.

The fixed capacity is what a site gets without forecasting: the service limit minus the building's
highest load in the training year.
"""
from __future__ import annotations

import numpy as np
from acnportal.acnsim.interface import InfrastructureInfo, SessionInfo
from adacharge import AdaptiveChargingOptimization, InfeasibilityException, ObjectiveComponent, equal_share, quick_charge

from ..sim.engine import StepView
from ..sim.scenario import STEP_H, Scenario


def _upper(view: StepView) -> np.ndarray:
    return np.minimum(view.max_kw, view.remaining_kwh / STEP_H)


def water_fill(upper: np.ndarray, budget: float) -> np.ndarray:
    """Equal share of `budget`, never above each car's own upper bound."""
    if upper.sum() <= budget:
        return upper.copy()
    out = np.zeros_like(upper)
    order = np.argsort(upper)
    left, n = budget, len(upper)
    for rank, i in enumerate(order):
        share = left / (n - rank)
        out[i] = min(upper[i], share)
        left -= out[i]
    return out


class Uncontrolled:
    name = "uncontrolled"

    def decide(self, view: StepView, scenario: Scenario) -> np.ndarray:
        return _upper(view)


class StaticCap:
    name = "static_cap"

    def __init__(self, cap_kw: float):
        self.cap_kw = cap_kw

    def decide(self, view: StepView, scenario: Scenario) -> np.ndarray:
        return water_fill(_upper(view), self.cap_kw)


class LeastLaxityFirst:
    """ACN-Sim's least-laxity-first rule: the car with the least slack charges first, at full power."""
    name = "llf"

    def __init__(self, cap_kw: float):
        self.cap_kw = cap_kw

    def decide(self, view: StepView, scenario: Scenario) -> np.ndarray:
        upper = _upper(view)
        laxity = (view.depart - view.t) - view.remaining_kwh / (view.max_kw * STEP_H)
        out = np.zeros_like(upper)
        left = self.cap_kw
        for i in np.argsort(laxity, kind="stable"):
            out[i] = min(upper[i], left)
            left -= out[i]
            if left <= 0:
                break
        return out


class _Interface:
    """The two attributes adacharge's constraints and objectives read from ACN-Sim's interface."""

    def __init__(self, period_min: float):
        self.period = period_min


class AdachargeMPC:
    """Caltech's adacharge MPC (quick_charge + equal_share, as in the ACN papers), run unmodified.

    Voltage is set to 1,000 V so one adacharge amp is one kW. Like ACN-Sim, it re-solves when the
    set of plugged-in cars changes and at least every `max_recompute` intervals, and otherwise
    follows its last plan.
    """
    name = "adacharge_mpc"

    def __init__(self, cap_kw: float, max_recompute: int = 4, solver: str = "CLARABEL"):
        self.cap_kw = cap_kw
        self.max_recompute = max_recompute
        self.optimizer = AdaptiveChargingOptimization(
            [ObjectiveComponent(quick_charge), ObjectiveComponent(equal_share, 1e-12)],
            interface=_Interface(STEP_H * 60), constraint_type="LINEAR", solver=solver)
        self._plan: dict[int, np.ndarray] = {}
        self._plan_t = -10**9
        self._plan_set: tuple = ()
        self.solves = 0
        self.failures = 0

    def _solve(self, view: StepView) -> None:
        stations = [f"s{i}" for i in view.active]
        n = len(stations)
        infra = InfrastructureInfo(
            constraint_matrix=np.ones((1, n)), constraint_limits=np.array([self.cap_kw]),
            phases=np.zeros(n), voltages=np.full(n, 1000.0), constraint_ids=["site"], station_ids=stations,
            max_pilot=view.max_kw.copy(), min_pilot=np.zeros(n), allowable_pilots=[np.array([0, m]) for m in view.max_kw],
            is_continuous=np.ones(n, dtype=bool))
        sessions = [SessionInfo(station_id=stations[j], session_id=str(i), requested_energy=float(view.remaining_kwh[j]),
                                energy_delivered=0.0, arrival=view.t, departure=int(view.depart[j]),
                                estimated_departure=int(view.depart[j]), current_time=view.t,
                                min_rates=0.0, max_rates=float(view.max_kw[j]))
                    for j, i in enumerate(view.active)]
        self.solves += 1
        try:
            rates = self.optimizer.solve(sessions, infra, peak_limit=self.cap_kw)
        except (InfeasibilityException, Exception):  # noqa: BLE001 - solver errors fall back to LLF
            self.failures += 1
            rates = LeastLaxityFirst(self.cap_kw).decide(view, None)[:, None]
        rates = np.nan_to_num(np.asarray(rates, dtype=float))
        self._plan = {int(i): rates[j] for j, i in enumerate(view.active)}
        self._plan_t = view.t
        self._plan_set = tuple(view.active)

    def decide(self, view: StepView, scenario: Scenario) -> np.ndarray:
        offset = view.t - self._plan_t
        if tuple(view.active) != self._plan_set or offset >= self.max_recompute:
            self._solve(view)
            offset = 0
        return np.array([self._plan[int(i)][offset] if offset < len(self._plan[int(i)]) else 0.0
                         for i in view.active])
