"""Score a simulation run against the building-only reference under a tariff."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sim.engine import Result, rise_above_recent_high
from .sim.scenario import STEP_H, Scenario
from .tariff import TariffSpec, monthly_bills

NEW_PEAK_TOL_KW = 0.5


def evaluate(result: Result, scenario: Scenario, spec: TariffSpec) -> dict:
    w = scenario.n_window
    index = scenario.index[:w]
    building = pd.Series(scenario.building_kw[:w], index=index)
    site = building + result.ev_kw[:w]
    with_ev = monthly_bills(site, spec)
    alone = monthly_bills(building, spec)
    demand_cols = [c for c in with_ev.columns if c.endswith("_kw")]

    new_peaks = int(((with_ev[demand_cols] - alone[demand_cols]) > NEW_PEAK_TOL_KW).to_numpy().sum())
    in_window = scenario.arrive < w
    request = scenario.request_kwh[in_window].sum()
    delivered = result.delivered_kwh[in_window].sum()
    short = scenario.request_kwh[in_window] - result.delivered_kwh[in_window]
    return {
        "controller": result.controller,
        "n_ports": scenario.n_ports,
        "sessions": int(in_window.sum()),
        "energy_served": delivered / request if request else 1.0,
        "cars_short_over_1kwh": float((short > 1.0).mean()) if in_window.any() else 0.0,
        "peak_site_kw": float(site.max()),
        "service_violations": int((site > scenario.service_kw + 1e-6).sum()),
        "new_demand_peaks": new_peaks,
        "added_demand_usd": float((with_ev["demand_usd"] - alone["demand_usd"]).sum()),
        "ev_energy_usd": float((with_ev["energy_usd"] - alone["energy_usd"]).sum()),
        "added_bill_usd": float((with_ev["total_usd"] - alone["total_usd"]).sum()),
        "ev_kwh": float(result.ev_kw[:w].sum() * STEP_H),
        "safety_interventions": int((result.safety_clipped_kw[:w] > 1e-6).sum()),
        "seconds": result.seconds,
    }


def ramp_breaches(scenario: Scenario, ramp_kw: float) -> int:
    """Intervals where the building rose above its recent high (sim.engine.HOLD) by more than the
    safety layer's assumed bound."""
    return int((rise_above_recent_high(scenario.building_kw) > ramp_kw).sum())
