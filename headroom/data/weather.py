"""Open-Meteo history for the building location, and its agreement with the BDG2 station.

Live mode forecasts come from Open-Meteo, so we measure how well its history tracks the station the
model trains on before deciding whether the two can be used interchangeably.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from ..config import RAW, SITE, SiteConfig

ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date={a}&end_date={b}&hourly=temperature_2m&timezone=GMT")
CACHE = RAW / "openmeteo_berkeley_2016_2017.csv"
INTERCHANGEABLE_R = 0.95


def openmeteo_history(site: SiteConfig = SITE) -> pd.Series:
    if not CACHE.exists():
        a = (site.start - pd.Timedelta("1D")).date()
        b = site.end.date()
        url = ARCHIVE.format(lat=site.weather_latlon[0], lon=site.weather_latlon[1], a=a, b=b)
        with urllib.request.urlopen(url, timeout=120) as resp:
            hourly = json.load(resp)["hourly"]
        pd.DataFrame({"ts_utc": hourly["time"], "temp_c": hourly["temperature_2m"]}).to_csv(CACHE, index=False)
    df = pd.read_csv(CACHE)
    return pd.Series(df["temp_c"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df["ts_utc"])).tz_localize("UTC"))


def compare_with_station(station_temp: pd.Series, site: SiteConfig = SITE) -> tuple[pd.Series, dict]:
    """Return Open-Meteo on the station's grid plus agreement statistics and the resulting decision."""
    model = openmeteo_history(site)
    both = pd.concat({"station": station_temp.resample("1h").mean(), "openmeteo": model}, axis=1).dropna()
    err = both["openmeteo"] - both["station"]
    r = float(both.corr().iloc[0, 1])
    stats = {
        "hours_compared": len(both),
        "pearson_r": round(r, 3),
        "bias_c": round(float(err.mean()), 2),
        "rmse_c": round(float(np.sqrt((err ** 2).mean())), 2),
        "decision": ("interchangeable: use Open-Meteo for training and live" if r > INTERCHANGEABLE_R else
                     "train on the station; calibrate live Open-Meteo forecasts to it"),
    }
    on_grid = model.reindex(model.index.union(station_temp.index)).interpolate("time").reindex(station_temp.index)
    return on_grid, stats
