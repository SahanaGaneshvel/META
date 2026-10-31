"""Headroom Check: how many chargers fit on any building's existing connection, from its own meter data.

Inputs: the building's interval meter data (hourly or 15-minute; a Green Button CSV export or any
timestamp + kW/kWh table), its service size (or, if unknown, the electrical-code rule that continuous
load stays within 80% of the rating: service = peak / 0.8), and a tariff. Charging demand is the real
Palo Alto sessions, resampled to the fleet size (headroom.sim.scenario).

The screening simulates whole months with each controller and finds each one's hosting capacity: the
largest fleet whose drivers get >= 99% of their energy in every simulated month with zero service-limit
violations. Standard smart charging (least-laxity-first) runs under a fixed cap of service minus the
building's highest load in the previous 12 months, the usual way a site sizes it; Headroom runs on the
hour-of-day profile forecast fitted on data before the window. Both run under the same safety layer,
whose allowance for the building's load rising above its past 3 hours' high is sized from that
building's own previous 12 months (largest rise + 20%); test-month rises beyond it are reported as
ramp breaches.

Timestamps are taken as local wall-clock and placed on the Palo Alto clock, so the building's 9 am
meets the drivers' 9 am (all buildings are billed on the same tariff, so results compare like-for-like).
"""
from __future__ import annotations

import dataclasses
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import INTERVAL, LOCAL_TZ, SITE
from .control.baselines import LeastLaxityFirst
from .control.headroom_mpc import HeadroomMPC
from .data.building import _on_hourly_grid, energy_preserving_quarter_hours
from .data.timeutil import wallclock_hourly_to_utc
from .forecast.profile import ProfileForecaster
from .metrics import evaluate, ramp_breaches
from .sim.engine import SafetyLayer, rise_above_recent_high, simulate
from .sim.scenario import make_scenario
from .tariff import TariffSpec

CONTINUOUS_LOAD_SHARE = 0.8  # NEC: continuous load within 80% of the service rating
SERVED_TARGET = 0.99
RAMP_MARGIN = 1.2  # safety layer allowance: the largest 15-minute rise of the past year, +20%


def ramp_allowance_kw(building_kw: pd.Series) -> float:
    """The safety layer's allowance for the building's load rising above its recent high, sized
    from this building's own history (each building ramps differently), rounded up to 5 kW."""
    rise = float(np.nanmax(rise_above_recent_high(building_kw.to_numpy()))) if len(building_kw) > 1 else 0.0
    return float(5 * np.ceil(max(rise, 1.0) * RAMP_MARGIN / 5))


# ---------------------------------------------------------------- meter data

def _read_table(source) -> pd.DataFrame:
    if isinstance(source, str) and "\n" in source:
        text = source
    elif isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8-sig", errors="replace")
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig", errors="replace")
    else:
        text = source.read()
        text = text.decode("utf-8-sig", errors="replace") if isinstance(text, bytes) else text
    lines = text.splitlines()
    # Green Button CSVs start with account lines; the table starts at the header naming the date column.
    start = next((i for i, line in enumerate(lines) if re.search(r"\b(date|time|timestamp)\b", line, re.I)), 0)
    return pd.read_csv(io.StringIO("\n".join(lines[start:])))


def load_meter(source) -> pd.Series:
    """Interval meter data -> average kW per interval, indexed by naive local wall-clock interval start.

    Accepts a timestamp column (or Green Button DATE + START TIME columns) and a usage column in kW
    (demand) or kWh (energy per interval; converted with the interval length)."""
    df = _read_table(source)
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    if "date" in df.columns and "start time" in df.columns:
        stamp = pd.to_datetime(df["date"].astype(str) + " " + df["start time"].astype(str))
    else:
        name = next(c for c in df.columns if re.search(r"time|date", c))
        stamp = pd.to_datetime(df[name])
    value_col = next((c for c in df.columns if re.search(r"\bkwh\b|usage|energy", c)), None)
    unit = "kwh"
    if value_col is None:
        value_col = next(c for c in df.columns if re.search(r"\bkw\b|demand|power|load", c))
        unit = "kw"
    values = pd.to_numeric(df[value_col].astype(str).str.replace(",", ""), errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(stamp)).sort_index()
    series = series[~series.index.duplicated()].dropna()
    step = pd.Series(series.index).diff().median()
    if step not in (pd.Timedelta("15min"), pd.Timedelta("1h")):
        raise ValueError(f"interval {step} not supported: export hourly or 15-minute data")
    if unit == "kwh":
        series = series / (step / pd.Timedelta("1h"))
    series.attrs["interval"] = "15min" if step == pd.Timedelta("15min") else "1h"
    return series


def site_table(meter_kw: pd.Series) -> pd.DataFrame:
    """Meter data -> the simulator's 15-minute UTC site table (building_kw), Palo Alto clock.

    15-minute data is used as measured (short gaps interpolated); hourly data gets the pipeline's
    energy-preserving split, so each hour's energy is unchanged."""
    first = pd.Timestamp(meter_kw.index[0].normalize(), tz=LOCAL_TZ).tz_convert("UTC")
    last = pd.Timestamp(meter_kw.index[-1].normalize() + pd.Timedelta(days=1), tz=LOCAL_TZ).tz_convert("UTC")
    grid = pd.date_range(first, last, freq=INTERVAL, inclusive="left", name="ts_utc")
    out = pd.DataFrame(index=grid)
    utc = wallclock_hourly_to_utc(meter_kw.clip(lower=0))  # works for any interval length
    if meter_kw.attrs.get("interval") == "15min":
        out["building_kw"] = utc.reindex(grid).interpolate(limit_direction="both").to_numpy()
    else:
        hours = pd.date_range(first, last, freq="1h", inclusive="left", name="ts_utc")
        load_h, _ = _on_hourly_grid(utc, hours)
        out["building_kw"] = energy_preserving_quarter_hours(load_h)
    return out


SESSIONS_END = pd.Timestamp("2017-12-01")  # last full month with real Palo Alto sessions starts here


def align_to_sessions(meter_kw: pd.Series) -> tuple[pd.Series, pd.Timedelta]:
    """Move a meter series from any year onto the 2016-2017 calendar of the real sessions in steps of
    52 weeks, so weekdays stay weekdays and seasons stay seasons (1 day of drift per year; holidays
    may land on different dates). Returns the moved series and the shift."""
    year = pd.Timedelta(days=364)
    steps = int(np.ceil((meter_kw.index[-1] - (SESSIONS_END + pd.Timedelta(days=30))) / year))
    shift = -steps * year
    moved = meter_kw.copy()
    moved.index = moved.index + shift
    moved.attrs = dict(meter_kw.attrs)
    return moved[moved.index >= pd.Timestamp("2016-01-01")], shift


def last_full_months(meter_kw: pd.Series, n: int = 2) -> list[str]:
    """The last n calendar months the data covers completely, with two months of history before."""
    step = pd.Timedelta(meter_kw.attrs.get("interval", "1h"))
    end = meter_kw.index[-1] + step  # end of the last interval
    last = end.to_period("M") - 1  # the month before `end`'s month is complete...
    if end == end.to_period("M").end_time.normalize() + pd.Timedelta(days=1) - pd.Timedelta(0):
        last = end.to_period("M")  # ...unless the data runs to the end of `end`'s own month
    months = [str(last - k) for k in range(n)][::-1]
    first_needed = pd.Timestamp(f"{months[0]}-01") - pd.Timedelta(days=60)
    if meter_kw.index[0] > first_needed:
        raise ValueError("need at least two months of history before the two months screened "
                         "(four months of interval data in total)")
    return months


# ---------------------------------------------------------------- screening

def _month_days(month: str) -> tuple[str, str]:
    first = pd.Timestamp(f"{month}-01")
    return str(first.date()), str((first + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date())


def run_size(site_df: pd.DataFrame, sessions: pd.DataFrame, spec: TariffSpec, service_kw: float,
             month: str, n_ports: int, controller: str, seed: int = 1) -> dict:
    first, last = _month_days(month)
    site = dataclasses.replace(SITE, service_limit_kw=service_kw)
    scenario = make_scenario(site_df, sessions, first, last, n_ports, seed=seed, site=site)
    start = pd.Timestamp(first, tz=LOCAL_TZ)
    past = site_df["building_kw"][(site_df.index < start.tz_convert("UTC"))
                                  & (site_df.index >= (start - pd.DateOffset(months=12)).tz_convert("UTC"))]
    if controller == "llf":
        ctrl = LeastLaxityFirst(max(service_kw - float(past.max()), 0.0))
    else:
        history = site_df["building_kw"]
        fit_start = str((start - pd.DateOffset(months=12)).date())
        forecaster = ProfileForecaster.fit(history, fit_start, first)
        ctrl = HeadroomMPC(forecaster, spec, history)
    ramp_kw = ramp_allowance_kw(past)
    row = evaluate(simulate(scenario, ctrl, SafetyLayer(service_kw, ramp_kw=ramp_kw)), scenario, spec)
    row.update(controller=controller, month=month, seed=seed, n_ports=n_ports, ramp_allowance_kw=ramp_kw,
               ramp_breaches=ramp_breaches(scenario, ramp_kw))
    return row


def _passes(rows: list[dict]) -> bool:
    return all(r["energy_served"] >= SERVED_TARGET and r["service_violations"] == 0 for r in rows)


def hosting_capacity(site_df, sessions, spec, service_kw, months, controller, seed=1,
                     start=2, max_ports=300, step=1.5, log=None) -> tuple[int, list[dict]]:
    """Exact largest fleet that passes in every month: grow geometrically until a size fails, then
    bisect between the last pass and the first fail. Returns it and every simulated month."""
    runs: list[dict] = []

    def ok(n: int) -> bool:
        rows = [run_size(site_df, sessions, spec, service_kw, m, n, controller, seed) for m in months]
        runs.extend(rows)
        if log:
            log(controller, n, rows)
        return _passes(rows)

    best, n = 0, start
    while n <= max_ports and ok(n):
        best, n = n, max(n + 1, int(round(n * step)))
    if n > max_ports:
        return best, runs  # capped: reported as ">= best"
    lo, hi = best, n  # lo passes (or 0), hi fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo, runs


def violations_at(runs: pd.DataFrame, llf_n: int, head_n: int) -> int:
    """Service-limit violations in the simulated months at the two capacities reported."""
    if not len(runs):
        return 0
    reported = (((runs["controller"] == "llf") & (runs["n_ports"] == llf_n))
                | ((runs["controller"] == "headroom") & (runs["n_ports"] == head_n)))
    return int(runs.loc[reported, "service_violations"].sum())


def check(site_df: pd.DataFrame, sessions: pd.DataFrame, spec: TariffSpec, months: list[str],
          service_kw: float | None = None, seed: int = 1, log=None,
          continuous_share: float = CONTINUOUS_LOAD_SHARE) -> dict:
    """The owner-facing answer for one building. Without a service size, it is estimated from the
    peak *before* the screened months (peak / continuous_share), so the test months aren't used."""
    before = site_df.index < pd.Timestamp(f"{months[0]}-01", tz=LOCAL_TZ).tz_convert("UTC")
    peak = float(site_df.loc[before, "building_kw"].max())
    assumed = service_kw is None
    service_kw = service_kw or round(peak / continuous_share, -1)
    llf_n, llf_runs = hosting_capacity(site_df, sessions, spec, service_kw, months, "llf", seed, log=log)
    head_n, head_runs = hosting_capacity(site_df, sessions, spec, service_kw, months, "headroom", seed,
                                         start=max(2, llf_n), log=log)
    runs = pd.DataFrame(llf_runs + head_runs)
    at = runs[runs["n_ports"] == llf_n] if llf_n else runs.iloc[0:0]
    same_fleet = at.groupby("controller")["added_demand_usd"].mean().to_dict() if len(at) else {}
    return {
        "building_peak_kw": round(peak), "service_kw": service_kw, "service_assumed_from_peak": assumed,
        "months": months, "chargers_standard_smart_charging": int(llf_n), "chargers_headroom": int(head_n),
        "extra_chargers": int(head_n - llf_n),
        "extra_share": (head_n - llf_n) / llf_n if llf_n else None,
        "added_demand_usd_per_month_at_standard_capacity": {k: round(v) for k, v in same_fleet.items()},
        "service_violations": int(runs["service_violations"].sum()),  # every size tried, rejected ones included
        "violations_at_reported_sizes": violations_at(runs, llf_n, head_n),
        "ramp_allowance_kw": float(runs["ramp_allowance_kw"].max()) if len(runs) else None,
        "ramp_breaches": int(runs.groupby("month")["ramp_breaches"].first().sum()) if len(runs) else 0,
        "runs": runs,
    }


def check_meter(meter_kw: pd.Series, sessions: pd.DataFrame, spec: TariffSpec,
                service_kw: float | None = None, log=None) -> dict:
    """Headroom Check for an owner's own meter export (any year): align, screen the last two full
    months, and report what was assumed."""
    moved, shift = align_to_sessions(meter_kw)
    months = last_full_months(moved)
    site_df = site_table(moved)
    out = check(site_df, sessions, spec, months, service_kw=service_kw, log=log)
    first = pd.Timestamp(f"{months[0]}-01") - shift
    out["your_months"] = f"{first:%b %Y} to {first + pd.DateOffset(months=len(months)) - pd.Timedelta(days=1):%b %Y}"
    out["calendar_shift_days"] = int(-shift.days)
    return out
