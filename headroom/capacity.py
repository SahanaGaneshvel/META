"""Hosting capacity: what happens if the site adds chargers, measured by simulation on real sessions.

For each charger count, every month is simulated with several resampled real-session fleets
(different seeds). The owner-facing answer for N chargers:
  - drivers served: share of months where >= 99% of requested energy is delivered;
  - added demand charges per month (mean and 95th percentile) vs. the building alone;
  - service-limit violations (must be 0).
Hosting capacity = the largest N where drivers are served in at least 99% of simulated months
with zero violations.
"""
from __future__ import annotations

import pandas as pd

SERVED_TARGET = 0.99


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """runs: one row per (controller, n_ports, month, seed) from metrics.evaluate."""
    g = runs.groupby(["controller", "n_ports"])
    out = pd.DataFrame({
        "scenarios": g.size(),
        "energy_served_mean": g["energy_served"].mean(),
        "energy_served_worst": g["energy_served"].min(),
        "months_drivers_served": g["energy_served"].apply(lambda s: (s >= SERVED_TARGET).mean()),
        "added_demand_usd_mean": g["added_demand_usd"].mean(),
        "added_demand_usd_p95": g["added_demand_usd"].quantile(0.95),
        "added_bill_usd_mean": g["added_bill_usd"].mean(),
        "service_violations": g["service_violations"].sum(),
        "peak_site_kw_max": g["peak_site_kw"].max(),
    })
    return out.reset_index()


def hosting_capacity(summary: pd.DataFrame, min_share: float = 0.99) -> pd.Series:
    ok = summary[(summary["months_drivers_served"] >= min_share) & (summary["service_violations"] == 0)]
    return ok.groupby("controller")["n_ports"].max()


def expansion_answer(summary: pd.DataFrame, n_ports: int, controller: str = "headroom",
                     reference: str = "llf") -> dict:
    """The facts the agent's simulate_expansion tool returns for one charger count."""
    table = summary.set_index(["controller", "n_ports"])
    available = sorted(summary.loc[summary["controller"] == controller, "n_ports"].unique())
    nearest = min(available, key=lambda n: abs(n - n_ports))
    row = table.loc[(controller, nearest)]
    ref = table.loc[(reference, nearest)] if (reference, nearest) in table.index else None
    answer = {
        "chargers_total": int(nearest),
        "simulated_months": int(row["scenarios"]),
        "drivers_fully_served_share_of_months": round(float(row["months_drivers_served"]), 3),
        "energy_served_mean": round(float(row["energy_served_mean"]), 4),
        "energy_served_worst_month": round(float(row["energy_served_worst"]), 4),
        "added_demand_charges_usd_per_month_mean": round(float(row["added_demand_usd_mean"])),
        "added_demand_charges_usd_per_month_p95": round(float(row["added_demand_usd_p95"])),
        "service_limit_violations": int(row["service_violations"]),
        "hosting_capacity_chargers": int(hosting_capacity(summary).get(controller, 0)),
    }
    if ref is not None:
        answer["standard_smart_charger_added_demand_usd_mean"] = round(float(ref["added_demand_usd_mean"]))
        answer["standard_smart_charger_energy_served_mean"] = round(float(ref["energy_served_mean"]), 4)
    if nearest != n_ports:
        answer["note"] = f"closest simulated size to the {n_ports} asked"
    return answer
