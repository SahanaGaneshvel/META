"""BDG2 building load and station weather, aligned to the site's 15-minute UTC grid.

BDG2 timestamps are local wall-clock: the weekday morning ramp of 78 Berkeley buildings sits at the
same clock hour in summer and winter (checked when choosing the building; see docs/DATA_SPEC.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from ..config import LOCAL_TZ, RAW, SITE, SiteConfig
from .timeutil import wallclock_hourly_to_utc

BDG2 = RAW / "bdg2"
SHORT_GAP_HOURS = 3


def _on_hourly_grid(series: pd.Series, hours: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Reindex to the full hourly grid; interpolate every gap, and mark hours inside long gaps."""
    s = series.reindex(hours)
    missing = s.isna()
    gap_length = missing.groupby((~missing).cumsum()).transform("sum")
    in_long_gap = missing & (gap_length > SHORT_GAP_HOURS)
    return s.interpolate(limit_direction="both"), in_long_gap


def energy_preserving_quarter_hours(hourly_kw: pd.Series) -> np.ndarray:
    """Hourly average kW -> 15-minute kW whose hourly means equal the input exactly.

    Interpolates cumulative energy with a monotone cubic (PCHIP) and differences it; monotone
    cumulative energy guarantees non-negative power.
    """
    cumulative = np.concatenate([[0.0], np.cumsum(hourly_kw.to_numpy())])
    curve = PchipInterpolator(np.arange(len(cumulative)), cumulative)
    quarter_marks = np.linspace(0, len(hourly_kw), 4 * len(hourly_kw) + 1)
    return np.diff(curve(quarter_marks)) * 4


def building_profile(site: SiteConfig = SITE) -> tuple[pd.DataFrame, dict]:
    meters = pd.read_csv(BDG2 / "electricity_cleaned.csv", usecols=["timestamp", site.building],
                         parse_dates=["timestamp"], index_col="timestamp")[site.building]
    weather = pd.read_csv(BDG2 / "weather.csv", parse_dates=["timestamp"])
    temperature = (weather[weather["site_id"] == site.weather_site]
                   .set_index("timestamp")["airTemperature"].sort_index())

    hours = site.grid("1h")
    load_h, long_gap = _on_hourly_grid(wallclock_hourly_to_utc(meters), hours)
    temp_h, _ = _on_hourly_grid(wallclock_hourly_to_utc(temperature), hours)

    grid = site.grid()
    out = pd.DataFrame(index=grid)
    out["building_kw"] = energy_preserving_quarter_hours(load_h)
    out["temp_c"] = temp_h.reindex(grid.union(hours)).interpolate("time").reindex(grid).to_numpy()
    gap_days = pd.Index(long_gap[long_gap].index.tz_convert(LOCAL_TZ).normalize().unique())
    out["imputed_day"] = grid.tz_convert(LOCAL_TZ).normalize().isin(gap_days)

    hourly_again = out["building_kw"].resample("1h").mean().to_numpy()
    stats = {
        "raw_hours_missing": int(meters.isna().sum()),
        "hours_in_long_gaps": int(long_gap.sum()),
        "imputed_days": len(gap_days),
        "hourly_energy_max_error_kw": float(np.abs(hourly_again - load_h.to_numpy()).max()),
        "building_kw_mean": round(out["building_kw"].mean(), 1),
        "building_kw_max": round(out["building_kw"].max(), 1),
        "temperature_coverage": round(temp_h.notna().mean(), 4),
    }
    return out, stats
