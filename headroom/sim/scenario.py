"""Simulation scenarios: a time window of real building load plus an N-port EV fleet.

N-port fleets are built by resampling real Webster port-days (docs/DATA_SPEC.md, section E): the
date's own 6 ports come first, so N = 6 reproduces the real garage exactly; extra ports draw
port-days with replacement from the same day type (workday or not) within +-7 days, shifted to
the target date at the same local clock time. Drawing from neighbouring dates, not only the same
date, keeps 60-port fleets from being ten copies of the same six cars.
"""
from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from ..config import INTERVAL, LOCAL_TZ, SITE, SiteConfig

STEP_H = INTERVAL / pd.Timedelta("1h")
POOL_DAYS = 7


@dataclass
class Scenario:
    index: pd.DatetimeIndex        # UTC interval starts; the window plus a tail for late departures
    n_window: int                  # intervals inside the evaluated window
    building_kw: np.ndarray        # actual building load (with intra-hour noise if requested)
    arrive: np.ndarray             # first interval each car can charge in
    depart: np.ndarray             # first interval it has gone (exclusive)
    request_kwh: np.ndarray        # energy the driver needs, capped at what the stay allows
    max_kw: np.ndarray
    port: np.ndarray
    n_ports: int
    service_kw: float
    label: str = ""

    @property
    def n_sessions(self) -> int:
        return len(self.arrive)


def _workdays(dates: pd.DatetimeIndex) -> np.ndarray:
    calendar = holidays.US(years=sorted(set(dates.year)))
    return np.array([d.weekday() < 5 and d not in calendar for d in dates.date])


def fleet_sessions(sessions: pd.DataFrame, first_day: str, last_day: str, n_ports: int,
                   seed: int, site: SiteConfig = SITE) -> pd.DataFrame:
    """Sessions for an n_ports garage arriving on local dates first_day..last_day (inclusive)."""
    rng = np.random.default_rng(seed)
    s = sessions[~sessions["failed_plugin"]].copy()
    s["local_day"] = s["arrival"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    ports = sorted(s["port_id"].unique())
    by_day = {day: g for day, g in s.groupby("local_day")}
    all_days = pd.DatetimeIndex(sorted(by_day))
    day_is_work = dict(zip(all_days, _workdays(all_days)))

    out = []
    for day in pd.date_range(first_day, last_day, freq="D"):
        pool = [d for d in all_days if abs((d - day).days) <= POOL_DAYS and day_is_work[d] == day_is_work.get(day, d.weekday() < 5)]
        picks = [(day, p) for p in ports] if day in by_day else []
        picks = picks[:n_ports]
        while len(picks) < n_ports:
            picks.append((pool[rng.integers(len(pool))], ports[rng.integers(len(ports))]))
        for virtual, (src_day, src_port) in enumerate(picks):
            g = by_day.get(src_day)
            if g is None:
                continue
            g = g[g["port_id"] == src_port]
            if g.empty:
                continue
            shift = day - src_day
            moved = g[["arrival", "departure", "kwh"]].copy()
            for col in ("arrival", "departure"):  # same local clock time on the target date
                local = moved[col].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None) + shift
                moved[col] = local.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
            moved["port"] = virtual
            out.append(moved.dropna())
    fleet = pd.concat(out, ignore_index=True).sort_values(["port", "arrival"])
    # A virtual port can receive overlapping stays from different source days: cut at the next arrival.
    next_arrival = fleet.groupby("port")["arrival"].shift(-1)
    fleet["departure"] = fleet["departure"].where(next_arrival.isna() | (fleet["departure"] <= next_arrival), next_arrival)
    return fleet.reset_index(drop=True)


def ar1_noise(n: int, sigma: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) with marginal std sigma. Stress tests only: headline runs use sigma = 0 (real data only)."""
    if sigma == 0:
        return np.zeros(n)
    e = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def make_scenario(site_df: pd.DataFrame, sessions: pd.DataFrame, first_day: str, last_day: str,
                  n_ports: int, seed: int = 0, noise_sigma: float = 0.0, noise_phi: float = 0.9,
                  tail_days: int = 1, site: SiteConfig = SITE) -> Scenario:
    start = pd.Timestamp(first_day, tz=LOCAL_TZ).tz_convert("UTC")
    stop = (pd.Timestamp(last_day, tz=LOCAL_TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
    tail_stop = stop + pd.Timedelta(days=tail_days)
    frame = site_df.loc[(site_df.index >= start) & (site_df.index < tail_stop)]
    index = frame.index
    rng = np.random.default_rng(seed + 10_000)
    building = frame["building_kw"].to_numpy() * (1 + ar1_noise(len(index), noise_sigma, noise_phi, rng))

    fleet = fleet_sessions(sessions, first_day, last_day, n_ports, seed, site)
    origin = index[0]
    arrive = np.ceil((fleet["arrival"] - origin) / INTERVAL).astype(int).to_numpy()
    depart = np.floor((fleet["departure"] - origin) / INTERVAL).astype(int).to_numpy()
    depart = np.minimum(depart, len(index))
    keep = depart > arrive
    max_kw = np.full(keep.sum(), site.port_kw_cap)
    request = np.minimum(fleet["kwh"].to_numpy()[keep], (depart[keep] - arrive[keep]) * max_kw * STEP_H)
    return Scenario(index=index, n_window=int(((index >= start) & (index < stop)).sum()), building_kw=building,
                    arrive=arrive[keep], depart=depart[keep], request_kwh=request, max_kw=max_kw,
                    port=fleet["port"].to_numpy()[keep], n_ports=n_ports, service_kw=site.service_limit_kw,
                    label=f"{first_day}..{last_day} N={n_ports} seed={seed}")
