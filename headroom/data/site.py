"""Join building, EV and weather data and add calendar and tariff-period features.

Tariff periods follow CPAU E-4 TOU (effective 2026-07-01): summer = May-Oct; weekday peak 16-21;
weekday mid-peak 14-16 and 21-23 in summer, 9-14 in winter; weekends and holidays off-peak.
Holidays = US federal (the CPAU definition still needs checking against its Rules & Regulations).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from ..config import LOCAL_TZ, SITE, SiteConfig


def assemble_site(building: pd.DataFrame, ev: pd.DataFrame, temp_openmeteo: pd.Series,
                  site: SiteConfig = SITE) -> tuple[pd.DataFrame, dict]:
    df = building.join(ev, how="inner")
    df["temp_c_openmeteo"] = temp_openmeteo.reindex(df.index).to_numpy()
    df["site_kw_uncontrolled"] = df["building_kw"] + df["ev_kw_uncontrolled"]

    local = df.index.tz_convert(LOCAL_TZ)
    day = local.tz_localize(None).normalize()
    clock = local.hour + local.minute / 60
    df["ts_local"] = local
    df["hour"] = local.hour
    df["dow"] = local.day_of_week
    calendar = holidays.US(years=sorted(set(day.year)))
    df["holiday"] = [calendar.get(d.date(), "") for d in day]

    workday = (local.day_of_week < 5) & (df["holiday"] == "")
    summer = local.month.isin(range(5, 11))
    df["season"] = np.where(summer, "summer", "winter")
    peak = workday & (clock >= 16) & (clock < 21)
    shoulder = np.where(summer, ((clock >= 14) & (clock < 16)) | ((clock >= 21) & (clock < 23)),
                        (clock >= 9) & (clock < 14))
    df["tou_period"] = np.select([peak, workday & shoulder], ["peak", "mid"], "off")
    df["peak_demand_window"] = peak

    df["flex_alert_day"] = False
    df["flex_alert_window"] = False
    for date, hours in site.flex_alerts.items():
        on = day == pd.Timestamp(date)
        start_h, end_h = hours or (14, 21)
        df.loc[on, "flex_alert_day"] = True
        df.loc[on & (clock >= start_h) & (clock < end_h), "flex_alert_window"] = True
    df["split"] = site.split_labels(local).to_numpy()

    by_split = df.groupby("split")
    stats = {
        "rows": len(df),
        "days_per_split": (by_split.size() * (df.index[1] - df.index[0]) / pd.Timedelta("1D")).round(1).to_dict(),
        "site_peak_kw_per_split": by_split["site_kw_uncontrolled"].max().round(1).to_dict(),
        "flex_alert_days_per_split": {k: sorted({str(t.date()) for t in g["ts_local"]})
                                      for k, g in df[df["flex_alert_day"]].groupby("split")},
        "holiday_days": int(df.loc[df["holiday"] != "", "ts_local"].dt.date.nunique()),
        "tou_share": df["tou_period"].value_counts(normalize=True).round(3).to_dict(),
    }
    return df, stats
