"""Timestamp parsing and localisation shared by the loaders."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ


def parse_export_dates(values: pd.Series) -> pd.Series:
    """Naive datetimes from the Palo Alto export, which mixes text dates with Excel serial numbers."""
    serial = pd.to_numeric(values, errors="coerce")
    is_serial = serial.notna()
    text = pd.to_datetime(values.where(~is_serial), format="mixed").dt.as_unit("s")
    excel = pd.to_datetime(serial, unit="D", origin="1899-12-30").dt.round("s").dt.as_unit("s")
    return text.where(~is_serial, excel)


def labelled_local_to_utc(naive: pd.Series, zone_label: pd.Series) -> pd.Series:
    """Wall-clock times tagged PDT/PST/UTC -> UTC. The tag decides the repeated hour in November."""
    labels = set(zone_label.unique())
    if not labels <= {"PDT", "PST", "UTC"}:
        raise ValueError(f"unexpected zone labels {labels - {'PDT', 'PST', 'UTC'}}")
    utc_rows = (zone_label == "UTC").to_numpy()
    as_local = naive[~utc_rows].dt.tz_localize(
        LOCAL_TZ, ambiguous=(zone_label[~utc_rows] == "PDT").to_numpy(), nonexistent="shift_forward")
    parts = [as_local.dt.tz_convert("UTC"), naive[utc_rows].dt.tz_localize("UTC")]
    return pd.concat(parts).reindex(naive.index)


def wallclock_hourly_to_utc(series: pd.Series) -> pd.Series:
    """Hourly wall-clock series with 24 rows on every day (BDG2 style) -> UTC-indexed series.

    The spring-forward 02:00 row cannot exist and is dropped; the single 01:00 row on the fall-back day
    is taken as the first (daylight) occurrence, leaving the repeated hour as a gap for the caller.
    """
    idx = series.index.tz_localize(LOCAL_TZ, ambiguous=np.ones(len(series), dtype=bool), nonexistent="NaT")
    out = pd.Series(series.to_numpy(), index=idx)[idx.notna()]
    out.index = out.index.tz_convert("UTC")
    return out[~out.index.duplicated()]
