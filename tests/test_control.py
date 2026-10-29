"""Invariants for the tariff, the simulator and the safety layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headroom.config import PROCESSED
from headroom.control.baselines import Uncontrolled, water_fill
from headroom.sim.engine import SafetyLayer, StepView, simulate
from headroom.sim.scenario import STEP_H, make_scenario
from headroom.tariff import load_gold, monthly_bills
from headroom.thermal import TransformerModel


@pytest.fixture(scope="module")
def data():
    return pd.read_parquet(PROCESSED / "site_15min.parquet"), pd.read_parquet(PROCESSED / "sessions.parquet")


def test_spike_costs_what_the_strategy_says():
    """One +50 kW, 15-minute spike at 6 pm on a summer weekday adds $2,572 to the month (E-4 TOU)."""
    spec = load_gold("e-4-tou_2026")
    index = pd.date_range("2017-07-01", "2017-08-01", freq="15min", tz="America/Los_Angeles", inclusive="left").tz_convert("UTC")
    base = pd.Series(100.0, index=index)
    spiked = base.copy()
    spiked[pd.Timestamp("2017-07-12 18:00", tz="America/Los_Angeles")] += 50
    extra = monthly_bills(spiked, spec)["total_usd"].iloc[0] - monthly_bills(base, spec)["total_usd"].iloc[0]
    assert extra == pytest.approx(50 * (30.18 + 21.27) + 50 * 0.25 * 0.21556, abs=0.01)


def test_six_ports_is_the_real_garage(data):
    site, sessions = data
    sc = make_scenario(site, sessions, "2016-08-01", "2016-08-07", n_ports=6, seed=3)
    real = sessions[~sessions["failed_plugin"]]
    local_day = real["arrival"].dt.tz_convert("America/Los_Angeles").dt.date.astype(str)
    expected = real[(local_day >= "2016-08-01") & (local_day <= "2016-08-07")]
    assert sc.n_sessions <= len(expected)
    assert sc.n_sessions >= 0.95 * len(expected)  # stays shorter than one interval can't charge
    assert set(np.unique(sc.port)) <= set(range(6))


def test_no_port_holds_two_cars(data):
    site, sessions = data
    sc = make_scenario(site, sessions, "2016-08-01", "2016-08-14", n_ports=36, seed=0)
    for port in np.unique(sc.port):
        rows = np.flatnonzero(sc.port == port)
        order = rows[np.argsort(sc.arrive[rows])]
        assert (sc.arrive[order][1:] >= sc.depart[order][:-1]).all()


def test_simulation_never_over_delivers(data):
    site, sessions = data
    sc = make_scenario(site, sessions, "2016-08-01", "2016-08-03", n_ports=24, seed=0)
    result = simulate(sc, Uncontrolled())
    assert (result.delivered_kwh <= sc.request_kwh + 1e-9).all()
    assert (result.delivered_kwh >= -1e-9).all()


def test_safety_layer_fuzz(data):
    """10,000 random and adversarial proposals: the output always respects port caps, remaining energy
    and the service limit, provided the building rose by no more than the layer's ramp bound."""
    site, sessions = data
    sc = make_scenario(site, sessions, "2016-08-01", "2016-08-03", n_ports=36, seed=0)
    layer = SafetyLayer(service_kw=sc.service_kw)
    rng = np.random.default_rng(0)
    bad_values = np.array([np.nan, np.inf, -np.inf, -1e9, 1e9])
    for trial in range(10_000):
        t = int(rng.integers(1, len(sc.index)))
        n = int(rng.integers(1, 40))
        active = rng.choice(sc.n_sessions, size=min(n, sc.n_sessions), replace=False)
        remaining = rng.uniform(0, 30, len(active))
        view = StepView(t=t, active=active, remaining_kwh=remaining, depart=sc.depart[active],
                        max_kw=sc.max_kw[active], building_measured=sc.building_kw[:t], ev_history=np.zeros(t))
        proposal = rng.normal(0, 50, len(active))
        if trial % 3 == 0:
            proposal[rng.integers(len(active))] = rng.choice(bad_values)
        if trial % 7 == 0:
            proposal = rng.normal(0, 50, len(active) + 1)  # wrong shape
        kw, _ = layer.apply(proposal, view, sc)
        assert kw.shape == active.shape and np.isfinite(kw).all()
        assert (kw >= 0).all() and (kw <= sc.max_kw[active] + 1e-9).all()
        assert (kw * STEP_H <= remaining + 1e-9).all()
        worst_building = sc.building_kw[t - 1] + layer.ramp_kw
        assert worst_building + kw.sum() <= sc.service_kw + 1e-6


def test_water_fill_is_fair_and_capped():
    out = water_fill(np.array([1.0, 5.0, 5.0]), 7.0)
    assert out == pytest.approx([1.0, 3.0, 3.0])
    assert water_fill(np.array([1.0, 2.0]), 10.0) == pytest.approx([1.0, 2.0])


def test_transformer_ages_at_the_reference_rate():
    """IEEE C57.91: a 110 C hot spot ages insulation at exactly 1 hour per hour."""
    model = TransformerModel()
    load = np.full(400, 1000.0)
    ambient = np.full(400, 110.0 - 55.0 - 25.0)  # rated rises bring the hot spot to 110 C
    assert model.hot_spot(load, ambient)[-1] == pytest.approx(110.0, abs=1e-6)
    assert model.loss_of_life_hours(load[-4:], ambient[-4:]) == pytest.approx(1.0, rel=1e-6)
