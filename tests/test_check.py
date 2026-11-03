"""Headroom Check reads the meter exports owners actually have, without changing their energy, and
refuses inputs it can't answer honestly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headroom.check import check, load_meter, site_table

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


def _table(stamps, values, header="timestamp,kW") -> str:
    return header + "\n" + "\n".join(f"{t},{v}" for t, v in zip(stamps, values))


def test_green_button_kwh_becomes_kw():
    kw = load_meter(GREEN_BUTTON.encode())
    assert kw.attrs["interval"] == "15min"
    assert kw.iloc[0] == 100.0 and kw.iloc[4] == 120.0  # 25 kWh in 15 min = 100 kW


def test_hourly_kw_table_keeps_hourly_energy():
    hours = pd.date_range("2017-07-01", periods=72, freq="1h")
    load = 200 + 100 * np.sin(np.arange(72) / 24 * 2 * np.pi)
    kw = load_meter(_table(hours, [f"{v:.3f}" for v in load]))
    assert kw.attrs["interval"] == "1h"
    table = site_table(kw)
    assert (table["building_kw"] >= 0).all()
    hourly_again = table["building_kw"].resample("1h").mean()
    utc_hours = hours.tz_localize("America/Los_Angeles").tz_convert("UTC")
    assert np.allclose(hourly_again.reindex(utc_hours).to_numpy(), load, atol=1e-3)


def test_utc_offsets_keep_the_local_clock():
    stamps = ["2017-03-11T23:00:00-08:00", "2017-03-12T00:00:00-08:00", "2017-03-12T01:00:00-08:00",
              "2017-03-12T03:00:00-07:00", "2017-03-12T04:00:00-07:00"]
    kw = load_meter(_table(stamps, [100, 110, 120, 130, 140]))
    assert kw.index[0] == pd.Timestamp("2017-03-11 23:00") and kw.index.tz is None
    assert kw.attrs["interval"] == "1h"


def test_watts_and_half_hours_are_converted():
    stamps = pd.date_range("2017-07-01", periods=6, freq="30min")
    kw = load_meter(_table(stamps, [50_000] * 6, header="time,Demand (W)"))
    assert kw.attrs["interval"] == "15min"
    assert (kw == 50.0).all() and len(kw) == 11  # each half hour fills both of its quarter hours


def test_five_minute_energy_is_averaged_to_quarter_hours():
    stamps = pd.date_range("2017-07-01", periods=6, freq="5min")
    kw = load_meter(_table(stamps, [1, 1, 1, 2, 2, 2], header="interval start,kWh"))
    assert kw.attrs["interval"] == "15min"
    assert kw.tolist() == [12.0, 24.0]  # 1 kWh per 5 min = 12 kW


def test_unreadable_tables_say_what_is_missing():
    with pytest.raises(ValueError, match="usage column"):
        load_meter("timestamp,notes\n2017-07-01 00:00,hello\n2017-07-01 01:00,there")
    with pytest.raises(ValueError, match="interval"):
        load_meter(_table(pd.date_range("2017-07-01", periods=4, freq="1D"), [1, 2, 3, 4]))


def _year_of(load_kw: float, gap_days: int = 0) -> pd.DataFrame:
    meter = pd.Series(load_kw, index=pd.date_range("2016-05-01", "2017-08-31 23:00", freq="1h"))
    if gap_days:
        meter = meter.drop(meter.index[24 * 100: 24 * (100 + gap_days)])  # Aug-Sep 2016, inside the year used
    meter.attrs["interval"] = "1h"
    return site_table(meter)


def test_a_connection_below_the_buildings_own_peak_is_refused():
    with pytest.raises(ValueError, match="not above this building's own peak"):
        check(_year_of(300.0), sessions=None, spec=None, months=["2017-07", "2017-08"], service_kw=250.0)


def test_meter_data_with_long_gaps_is_refused():
    with pytest.raises(ValueError, match="gaps longer than 3 hours"):
        check(_year_of(300.0, gap_days=40), sessions=None, spec=None, months=["2017-07", "2017-08"])
