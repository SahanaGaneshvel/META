"""Headroom's planner: building-aware MPC on the forecast's quantiles, solved as an LP with HiGHS.

Every interval it plans all plugged-in cars to their departures (at most 24 h ahead):

    min   energy cost + unserved_usd_per_kwh * unserved + demand rates * ceiling raises
    s.t.  each car: delivered + unserved = remaining need;  0 <= kW <= port cap while plugged in
          EV kW at k <= max(0, ceiling - building P_hi forecast at k) + raise        (max demand)
          same for 4-9 pm weekdays against the peak-period ceiling                  (peak demand)
          EV kW at k <= max(0, service limit - building P_hi forecast at k)

Ceilings are what the month has already paid for (month-to-date site peaks), floored at levels
the building alone will very probably reach anyway, so filling up to them is almost certainly free:
  - the lowest monthly building peak of the same season in the previous 12 months (past data only),
  - the building's low-quantile forecast peak over the next day.
The forecast quantiles (q_low for floors, q_high for the building under the EVs) are the owner's
risk dial; the Monte Carlo in `headroom.capacity` measures the risk they actually buy. Only the first interval of the plan is used; the safety layer still has the last word.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

from ..config import LOCAL_TZ
from ..sim.engine import StepView
from ..sim.scenario import STEP_H, Scenario
from ..tariff import TariffSpec, interval_labels
from .baselines import LeastLaxityFirst


class HeadroomMPC:
    name = "headroom"

    def __init__(self, forecaster, spec: TariffSpec, history: pd.Series, q_low: float = 0.1,
                 q_high: float = 0.99, unserved_usd_per_kwh: float = 2.0, horizon: int = 96):
        self.forecaster = forecaster
        self.history = history
        self.spec = spec
        self.q_low = forecaster.quantile_levels.index(q_low)
        self.q_high = forecaster.quantile_levels.index(q_high)
        self.unserved = unserved_usd_per_kwh
        self.horizon = horizon
        self._scenario_id = None
        self.failures = 0

    def _prepare(self, scenario: Scenario) -> None:
        labels = interval_labels(scenario.index, self.spec)
        self.price = labels["usd_per_kwh"].to_numpy()
        peak_cols = [c for c in labels.columns if c.startswith("demand:") and c != "demand:max demand"]
        self.peak_window = labels[peak_cols].any(axis=1).to_numpy() if peak_cols else np.zeros(len(labels), bool)
        local = scenario.index.tz_convert(LOCAL_TZ)
        self.month = np.asarray(local.year * 12 + local.month)
        rates = {}
        for season in self.spec.seasons:
            for charge in season.demand:
                rates[(season.name, "max" if not charge.windows else "peak")] = charge.usd_per_kw
        seasons = labels["season"].to_numpy()
        self.rate_max = np.array([rates.get((s, "max"), 0.0) for s in seasons])
        self.rate_peak = np.array([rates.get((s, "peak"), 0.0) for s in seasons])
        self.floors = self._month_floors(local, labels, seasons)
        self.forecast = None  # filled lazily: needs measured data, which grows as the run proceeds
        self._scenario_id = id(scenario)
        self._mtd = {"month": None, "max": 0.0, "peak": 0.0}

    def _month_floors(self, local: pd.DatetimeIndex, labels: pd.DataFrame, seasons: np.ndarray) -> dict:
        """(month key) -> (max-demand floor, peak-period floor) from the 12 months before that month."""
        h = self.history
        h_labels = interval_labels(h.index, self.spec)
        peak_cols = [c for c in h_labels.columns if c.startswith("demand:") and c != "demand:max demand"]
        h_peak = h_labels[peak_cols].any(axis=1)
        h_local = h.index.tz_convert(LOCAL_TZ)
        h_key = np.asarray(h_local.year * 12 + h_local.month)
        monthly = pd.DataFrame({"key": h_key, "kw": h.to_numpy(), "peak_kw": h.where(h_peak).to_numpy(),
                                "season": h_labels["season"].to_numpy()}).groupby("key").agg(
            kw=("kw", "max"), peak_kw=("peak_kw", "max"), season=("season", "first"))
        floors = {}
        for key, season in zip(self.month, seasons):
            if key in floors:
                continue
            past = monthly[(monthly.index < key) & (monthly.index >= key - 12) & (monthly["season"] == season)]
            floors[key] = (float(past["kw"].min()) if len(past) else 0.0,
                           float(past["peak_kw"].min()) if len(past) and past["peak_kw"].notna().any() else 0.0)
        return floors

    def _quantiles(self, t: int, scenario: Scenario) -> np.ndarray:
        # The forecaster only uses measured[:t] for row t, so passing the full array is leak-free;
        # it's computed once per scenario for speed.
        if self.forecast is None:
            self.forecast = self.forecaster.predict(scenario.index, scenario.building_kw)
        return self.forecast[t]

    def _update_month_to_date(self, view: StepView) -> None:
        t = view.t
        if self._mtd["month"] != self.month[t]:
            self._mtd = {"month": self.month[t], "max": 0.0, "peak": 0.0}
        if t > 0 and self.month[t - 1] == self.month[t]:
            site = view.building_measured[-1] + view.ev_history[-1]
            self._mtd["max"] = max(self._mtd["max"], site)
            if self.peak_window[t - 1]:
                self._mtd["peak"] = max(self._mtd["peak"], site)

    def decide(self, view: StepView, scenario: Scenario) -> np.ndarray:
        if self._scenario_id != id(scenario):
            self._prepare(scenario)
        self._update_month_to_date(view)
        t, n = view.t, len(view.active)
        H = int(min(self.horizon, max(1, view.depart.max() - t), len(scenario.index) - t))
        q = self._quantiles(t, scenario)[:H]
        b_low, b_high = q[:, self.q_low], q[:, self.q_high]
        same_month = self.month[t:t + H] == self.month[t]
        peak = self.peak_window[t:t + H] & same_month
        floor_max, floor_peak = self.floors[self.month[t]]
        ceiling_max = max(self._mtd["max"], floor_max, b_low[same_month].max())
        ceiling_peak = max(self._mtd["peak"], floor_peak, b_low[peak].max() if peak.any() else 0.0)

        # Variables: x[i, k] (n*H, row-major), unserved[i] (n), raise_max, raise_peak.
        nx = n * H
        steps = np.arange(H)
        plugged = (t + steps[None, :]) < view.depart[:, None]
        upper_x = np.where(plugged, view.max_kw[:, None], 0.0).ravel()
        bounds = [(0.0, u) for u in upper_x] + [(0.0, r) for r in view.remaining_kwh] + [(0.0, None), (0.0, None)]

        c = np.concatenate([
            (np.tile(self.price[t:t + H], n) - 1e-3 * np.tile(1 - steps / H, n)) * STEP_H,
            np.full(n, self.unserved),
            [self.rate_max[t], self.rate_peak[t] if peak.any() else 0.0]])

        rows_i = np.repeat(np.arange(n), H)
        cols = np.arange(nx)
        A_eq = sp.csr_matrix((np.concatenate([np.full(nx, STEP_H), np.ones(n)]),
                              (np.concatenate([rows_i, np.arange(n)]), np.concatenate([cols, nx + np.arange(n)]))),
                             shape=(n, nx + n + 2))
        b_eq = view.remaining_kwh

        k_of = np.tile(steps, n)
        blocks, rhs = [], []
        # max-demand ceiling, all steps of this month
        sel = same_month[k_of]
        m_rows = np.flatnonzero(same_month)
        row_map = -np.ones(H, int)
        row_map[m_rows] = np.arange(len(m_rows))
        blocks.append(sp.csr_matrix(
            (np.concatenate([np.ones(sel.sum()), -np.ones(len(m_rows))]),
             (np.concatenate([row_map[k_of[sel]], np.arange(len(m_rows))]),
              np.concatenate([cols[sel], np.full(len(m_rows), nx + n)]))), shape=(len(m_rows), nx + n + 2)))
        rhs.append(np.maximum(0.0, ceiling_max - b_high[m_rows]))
        # peak-period ceiling
        if peak.any():
            p_rows = np.flatnonzero(peak)
            row_map = -np.ones(H, int)
            row_map[p_rows] = np.arange(len(p_rows))
            sel = peak[k_of]
            blocks.append(sp.csr_matrix(
                (np.concatenate([np.ones(sel.sum()), -np.ones(len(p_rows))]),
                 (np.concatenate([row_map[k_of[sel]], np.arange(len(p_rows))]),
                  np.concatenate([cols[sel], np.full(len(p_rows), nx + n + 1)]))), shape=(len(p_rows), nx + n + 2)))
            rhs.append(np.maximum(0.0, ceiling_peak - b_high[p_rows]))
        # service limit on the high forecast
        blocks.append(sp.csr_matrix((np.ones(nx), (k_of, cols)), shape=(H, nx + n + 2)))
        rhs.append(np.maximum(0.0, scenario.service_kw - b_high))

        res = linprog(c, A_ub=sp.vstack(blocks).tocsr(), b_ub=np.concatenate(rhs), A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.status != 0:
            self.failures += 1
            return LeastLaxityFirst(max(0.0, ceiling_max - b_high[0])).decide(view, scenario)
        return res.x[:nx].reshape(n, H)[:, 0]
