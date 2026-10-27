"""Data invariants. Unit tests run anywhere; table tests need `python -m headroom.data.pipeline` first."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headroom.config import PROCESSED, SITE
from headroom.data.building import energy_preserving_quarter_hours
from headroom.data.timeutil import labelled_local_to_utc, parse_export_dates, wallclock_hourly_to_utc


# ---------------------------------------------------------------- helpers
def test_excel_serial_dates_are_parsed():
    out = parse_export_dates(pd.Series(["7/29/2011 20:17", "42845.81041"]))
    assert out[0] == pd.Timestamp("2011-07-29 20:17")
    assert out[1].floor("min") == pd.Timestamp("2017-04-20 19:26")


def test_fall_back_hour_uses_zone_label():
    naive = pd.Series(pd.to_datetime(["2016-11-06 01:30", "2016-11-06 01:30"]))
    utc = labelled_local_to_utc(naive, pd.Series(["PDT", "PST"]))
    assert (utc[1] - utc[0]) == pd.Timedelta("1h")


def test_bdg2_dst_days_have_no_duplicate_or_impossible_hours():
    idx = pd.date_range("2016-03-13", periods=24, freq="1h")
    out = wallclock_hourly_to_utc(pd.Series(np.arange(24.0), index=idx))
    assert len(out) == 23 and out.index.is_unique


def test_quarter_hours_preserve_hourly_energy_and_stay_positive():
    hourly = pd.Series([600.0, 820.0, 610.0, 590.0, 700.0])
    q = energy_preserving_quarter_hours(hourly)
    assert np.allclose(q.reshape(-1, 4).mean(axis=1), hourly)
    assert (q >= 0).all()


# ---------------------------------------------------------------- processed tables
@pytest.fixture(scope="module")
def site():
    path = PROCESSED / "site_15min.parquet"
    if not path.exists():
        pytest.skip("run `python -m headroom.data.pipeline` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def sessions():
    path = PROCESSED / "sessions.parquet"
    if not path.exists():
        pytest.skip("run `python -m headroom.data.pipeline` first")
    return pd.read_parquet(path)


def test_grid_is_complete_and_uniform(site):
    assert len(site) == len(SITE.grid())
    assert site.index.is_unique and (site.index.to_series().diff().iloc[1:] == pd.Timedelta("15min")).all()


def test_physical_limits(site, sessions):
    ports = sessions["port_id"].nunique()
    assert site["ev_plugged"].max() <= ports + 1e-9
    assert site["ev_kw_uncontrolled"].max() <= ports * SITE.port_kw_cap + 1e-6
    assert (site["building_kw"] > 0).all()
    assert (site["ev_idle"] <= site["ev_plugged"] + 1e-9).all()


def test_no_port_holds_two_cars(sessions):
    s = sessions.sort_values(["port_id", "arrival"])
    prev_departure = s.groupby("port_id")["departure"].shift()
    assert not (s["arrival"] < prev_departure - pd.Timedelta("1min")).any()


def test_splits_are_time_ordered_and_disjoint(site):
    order = {"train": 0, "val": 1, "test": 2}
    ranks = site["split"].map(order)
    assert ranks.notna().all() and ranks.is_monotonic_increasing


def test_demo_day_has_real_peak_window_shock(site):
    window = site[site["peak_demand_window"]].set_index("ts_local")["building_kw"]
    demo = window.loc["2017-06-21"].max()
    week_before = window.loc["2017-06-14"].max()
    assert demo > 1.05 * week_before  # Flex Alert heat day: building 4-9 pm peak clearly higher
