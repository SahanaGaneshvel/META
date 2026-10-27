"""Sessions -> time-weighted 15-minute EV occupancy and uncontrolled charging load."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import INTERVAL, SITE, SiteConfig

MINUTE = pd.Timedelta("1min")


def ev_profile(sessions: pd.DataFrame, site: SiteConfig = SITE) -> tuple[pd.DataFrame, dict]:
    """Integrate each session on a 1-minute grid, then average to intervals.

    Uncontrolled charging = each car draws its measured average power from arrival until its energy
    is delivered, then sits idle until departure.
    """
    origin = site.start.tz_convert("UTC")
    minutes = int((site.end - site.start) / MINUTE)
    per_interval = int(INTERVAL / MINUTE)

    def minute_of(ts: pd.Series) -> np.ndarray:
        return np.clip(((ts - origin) / MINUTE).round().astype(int).to_numpy(), 0, minutes)

    t_arrive = minute_of(sessions["arrival"])
    t_leave = minute_of(sessions["departure"])
    t_full = minute_of(sessions["arrival"] + pd.to_timedelta(sessions["charging_h"], unit="h"))

    def integrate(starts: np.ndarray, stops: np.ndarray, level: np.ndarray) -> np.ndarray:
        delta = np.zeros(minutes + 1)
        np.add.at(delta, starts, level)
        np.add.at(delta, stops, -level)
        return np.cumsum(delta)[:minutes].reshape(-1, per_interval).mean(axis=1)

    charging = ~sessions["failed_plugin"].to_numpy()
    power = sessions["avg_kw"].to_numpy()
    grid = site.grid()
    slot = t_arrive // per_interval
    in_range = slot < len(grid)

    profile = pd.DataFrame(index=grid)
    profile["ev_plugged"] = integrate(t_arrive, t_leave, np.ones(len(sessions)))
    profile["ev_idle"] = integrate(t_full, t_leave, np.ones(len(sessions)))
    profile["ev_kw_uncontrolled"] = integrate(t_arrive[charging], t_full[charging], power[charging])
    profile["ev_arrivals"] = np.bincount(slot[in_range], minlength=len(grid))
    profile["ev_kwh_arriving"] = np.bincount(slot[in_range], weights=sessions["kwh"].to_numpy()[in_range],
                                             minlength=len(grid))

    delivered = profile["ev_kw_uncontrolled"].sum() * (INTERVAL / pd.Timedelta("1h"))
    requested = sessions.loc[charging, "kwh"].sum()
    stats = {
        "intervals": len(profile),
        "energy_balance_error": round(abs(delivered - requested) / requested, 5),
        "max_cars_plugged": round(profile["ev_plugged"].max(), 2),
        "peak_uncontrolled_kw": round(profile["ev_kw_uncontrolled"].max(), 1),
        "idle_share_of_plugged_time": round(profile["ev_idle"].sum() / profile["ev_plugged"].sum(), 3),
    }
    return profile, stats
