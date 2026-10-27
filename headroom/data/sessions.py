"""Palo Alto charging sessions for the site's garage, cleaned to physically consistent records."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ, RAW, SITE, SiteConfig
from .timeutil import labelled_local_to_utc, parse_export_dates

SOURCE = RAW / "paloalto_ev_sessions.csv"
USECOLS = {
    "Station Name": "station",
    "Port Number": "port",
    "Start Date": "start_raw",
    "Start Time Zone": "start_zone",
    "End Date": "end_raw",
    "End Time Zone": "end_zone",
    "Charging Time (hh:mm:ss)": "charging_raw",
    "Energy (kWh)": "kwh",
    "User ID": "user_id",
}
FAILED_PLUGIN_KWH = 0.1


def _hours(hhmmss: pd.Series) -> pd.Series:
    h, m, s = (hhmmss.astype(str).str.split(":", expand=True).astype(float)[i] for i in range(3))
    return h + m / 60 + s / 3600


def load_sessions(site: SiteConfig = SITE) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(SOURCE, usecols=list(USECOLS), low_memory=False).rename(columns=USECOLS)
    df = raw[raw["station"].str.contains(site.garage, na=False)].copy()

    df["arrival"] = labelled_local_to_utc(parse_export_dates(df["start_raw"]), df["start_zone"])
    df["departure"] = labelled_local_to_utc(parse_export_dates(df["end_raw"]), df["end_zone"])
    arrival_local = df["arrival"].dt.tz_convert(LOCAL_TZ)
    df = df[(arrival_local >= site.start) & (arrival_local < site.end)].copy()
    stats: dict = {"sessions_in_period": len(df), "zone_labels": df["start_zone"].value_counts().to_dict()}

    df["port_id"] = df["station"].str.extract(r"(#\d+)", expand=False) + "-P" + df["port"].astype(str)
    df["kwh"] = df["kwh"].astype(float)
    df["charging_h"] = _hours(df["charging_raw"])
    stay_h = (df["departure"] - df["arrival"]).dt.total_seconds() / 3600

    rules = {
        "ends_before_start": stay_h <= 0,
        "stay_over_24h": stay_h > 24,
        "charging_longer_than_stay": df["charging_h"] > stay_h + 0.1,
        "duplicate_record": df.duplicated(["port_id", "arrival"]),
    }
    keep = pd.Series(True, index=df.index)
    for name, broken in rules.items():
        stats[f"removed_{name}"] = int((broken & keep).sum())
        keep &= ~broken
    df = df[keep].sort_values(["port_id", "arrival"]).copy()

    # One car per port: a record still open when the next car plugs in was never closed in the log.
    later_arrival = df.groupby("port_id")["arrival"].shift(-1)
    ran_into_next = later_arrival.notna() & (later_arrival < df["departure"])
    stats["stays_cut_at_next_arrival"] = int(ran_into_next.sum())
    df.loc[ran_into_next, "departure"] = later_arrival[ran_into_next]
    df["stay_h"] = (df["departure"] - df["arrival"]).dt.total_seconds() / 3600

    # Charging must fit the port's power cap and the stay.
    minimum_h = df["kwh"] / site.port_kw_cap
    stats["charging_time_raised_to_power_cap"] = int((df["charging_h"] < minimum_h - 1e-9).sum())
    df["charging_h"] = df["charging_h"].clip(lower=minimum_h).clip(upper=df["stay_h"])
    df["avg_kw"] = (df["kwh"] / df["charging_h"].replace(0, np.nan)).fillna(0.0)
    df["idle_h"] = df["stay_h"] - df["charging_h"]
    df["failed_plugin"] = df["kwh"] < FAILED_PLUGIN_KWH  # occupied a bay, delivered ~nothing
    df["split"] = site.split_labels(df["arrival"].dt.tz_convert(LOCAL_TZ)).to_numpy()

    out = df[["port_id", "arrival", "departure", "stay_h", "charging_h", "idle_h", "kwh", "avg_kw",
              "failed_plugin", "user_id", "split"]].reset_index(drop=True)
    stats.update(
        sessions_kept=len(out),
        ports=sorted(out["port_id"].unique()),
        energy_kwh=round(out["kwh"].sum(), 1),
        failed_plugins=int(out["failed_plugin"].sum()),
        idle_h_median=round(out["idle_h"].median(), 2),
        idle_h_p90=round(out["idle_h"].quantile(0.9), 2),
        sessions_per_split=out["split"].value_counts().to_dict(),
    )
    return out, stats
