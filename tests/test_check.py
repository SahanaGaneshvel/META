"""Headroom Check reads the meter exports owners actually have, without changing their energy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from headroom.check import load_meter, site_table

GREEN_BUTTON = """Name,ACME MALL
Address,1 Main St
Account Number,1234

TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES
Electric usage,2017-07-01,00:00,00:14,25.00,$3.10,
Electric usage,2017-07-01,00:15,00:29,26.00,$3.22,
Electric usage,2017-07-01,00:30,00:44,24.00,$2.98,
Electric usage,2017-07-01,00:45,00:59,25.00,$3.10,
Electric usage,2017-07-01,01:00,01:14,30.00,$3.72,
"""


def test_green_button_kwh_becomes_kw():
    kw = load_meter(GREEN_BUTTON.encode())
    assert kw.attrs["interval"] == "15min"
    assert kw.iloc[0] == 100.0 and kw.iloc[4] == 120.0  # 25 kWh in 15 min = 100 kW


def test_hourly_kw_table_keeps_hourly_energy():
    hours = pd.date_range("2017-07-01", periods=72, freq="1h")
    load = 200 + 100 * np.sin(np.arange(72) / 24 * 2 * np.pi)
    text = "timestamp,kW\n" + "\n".join(f"{t},{v:.3f}" for t, v in zip(hours, load))
    kw = load_meter(text)
    assert kw.attrs["interval"] == "1h"
    table = site_table(kw)
    assert (table["building_kw"] >= 0).all()
    hourly_again = table["building_kw"].resample("1h").mean()
    utc_hours = hours.tz_localize("America/Los_Angeles").tz_convert("UTC")
    assert np.allclose(hourly_again.reindex(utc_hours).to_numpy(), load, atol=1e-3)
