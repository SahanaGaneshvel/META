"""Utility tariffs as data, and the monthly bill they produce.

`TariffSpec` is both the hand-checked gold format (`data/docs/gold/`) and the strict JSON schema the
operator agent fills in when it reads a tariff document; `monthly_bills` is the verifier that checks
an extraction by reproducing a bill.
"""
from __future__ import annotations

from typing import Literal

import holidays
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import DATA, LOCAL_TZ

GOLD = DATA / "docs" / "gold"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Window(_Strict):
    days: Literal["weekdays", "weekends", "all"] = Field(
        description="weekdays = Monday-Friday excluding holidays when the tariff says so")
    start_hour: float = Field(ge=0, le=24, description="local clock hour the window opens, e.g. 16 for 4:00 pm")
    end_hour: float = Field(ge=0, le=24, description="local clock hour the window closes (exclusive)")


class EnergyRate(_Strict):
    period: str = Field(description="period name as printed, e.g. 'peak', 'mid-peak'")
    usd_per_kwh: float = Field(ge=0, description="total rate across all unbundled components")
    windows: list[Window] = Field(description="empty for the period that covers all remaining hours")


class DemandCharge(_Strict):
    name: str = Field(description="e.g. 'peak demand', 'max demand'")
    usd_per_kw: float = Field(ge=0, description="total rate across all unbundled components")
    windows: list[Window] = Field(description="hours whose 15-minute demand counts; empty = all hours")


class Season(_Strict):
    name: str
    months: list[int] = Field(description="calendar months 1-12 in this season")
    energy: list[EnergyRate]
    demand: list[DemandCharge]


class TariffSpec(_Strict):
    utility: str
    schedule: str
    effective: str = Field(description="ISO date the rates take effect")
    demand_interval_minutes: int = Field(description="length of the interval whose average kW is billed")
    holidays_off_peak: bool = Field(description="True if the tariff treats holidays like weekends")
    customer_charge_usd_per_month: float = Field(ge=0)
    seasons: list[Season]

    def season_of(self, month: int) -> Season:
        for season in self.seasons:
            if month in season.months:
                return season
        raise ValueError(f"no season covers month {month}")


def load_gold(name: str) -> TariffSpec:
    return TariffSpec.model_validate_json((GOLD / f"{name}.json").read_text(encoding="utf-8"))


def _in_windows(windows: list[Window], clock: np.ndarray, workday: np.ndarray) -> np.ndarray:
    hit = np.zeros(len(clock), dtype=bool)
    for w in windows:
        days = {"weekdays": workday, "weekends": ~workday, "all": np.ones(len(clock), dtype=bool)}[w.days]
        hit |= days & (clock >= w.start_hour) & (clock < w.end_hour)
    return hit


def interval_labels(index: pd.DatetimeIndex, spec: TariffSpec) -> pd.DataFrame:
    """Per interval: season, energy period, price, and one boolean column per demand charge."""
    local = index.tz_convert(LOCAL_TZ)
    clock = np.asarray(local.hour + local.minute / 60, dtype=float)
    workday = np.asarray(local.day_of_week < 5)
    if spec.holidays_off_peak:
        calendar = holidays.US(years=sorted(set(local.year)))
        workday &= ~np.array([d in calendar for d in local.date])
    months = np.asarray(local.month)

    out = pd.DataFrame(index=index)
    out["season"] = ""
    out["period"] = ""
    out["usd_per_kwh"] = np.nan
    demand_names = sorted({d.name for s in spec.seasons for d in s.demand})
    for name in demand_names:
        out[f"demand:{name}"] = False
    for season in spec.seasons:
        rows = np.isin(months, season.months)
        out.loc[rows, "season"] = season.name
        unassigned = rows.copy()
        for rate in sorted(season.energy, key=lambda r: not r.windows):  # windowed periods first
            hit = unassigned & (_in_windows(rate.windows, clock, workday) if rate.windows else True)
            out.loc[hit, "period"] = rate.period
            out.loc[hit, "usd_per_kwh"] = rate.usd_per_kwh
            unassigned &= ~hit
        for charge in season.demand:
            hit = rows & (_in_windows(charge.windows, clock, workday) if charge.windows else True)
            out.loc[hit, f"demand:{charge.name}"] = True
    return out


def monthly_bills(site_kw: pd.Series, spec: TariffSpec) -> pd.DataFrame:
    """Bill per calendar month for a site load series (kW, UTC index on the demand interval).

    Billing periods are calendar months. The CPAU seasons start on May 1 and Nov 1, so no month
    straddles a season and the proration rule never applies.
    """
    step_h = (site_kw.index[1] - site_kw.index[0]) / pd.Timedelta("1h")
    if abs(step_h * 60 - spec.demand_interval_minutes) > 1e-9:
        raise ValueError("site_kw must be sampled on the tariff's demand interval")
    labels = interval_labels(site_kw.index, spec)
    month = site_kw.index.tz_convert(LOCAL_TZ).strftime("%Y-%m")
    frame = labels.assign(kw=site_kw.to_numpy(), month=month)
    frame["energy_usd"] = frame["kw"] * step_h * frame["usd_per_kwh"]

    rows = []
    for m, g in frame.groupby("month"):
        season = spec.season_of(int(m[-2:]))
        row = {"month": m, "season": season.name, "kwh": g["kw"].sum() * step_h,
               "energy_usd": g["energy_usd"].sum(), "customer_usd": spec.customer_charge_usd_per_month}
        demand_usd = 0.0
        for charge in season.demand:
            mask = g[f"demand:{charge.name}"]
            peak = float(g.loc[mask, "kw"].max()) if mask.any() else 0.0
            row[f"{charge.name}_kw"] = peak
            row[f"{charge.name}_usd"] = peak * charge.usd_per_kw
            demand_usd += peak * charge.usd_per_kw
        row["demand_usd"] = demand_usd
        row["total_usd"] = row["energy_usd"] + demand_usd + row["customer_usd"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")
