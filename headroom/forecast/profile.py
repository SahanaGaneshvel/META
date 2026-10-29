"""Hour-of-day profile forecaster with quantiles: the baseline layer of the forecasting stack.

Point forecast for interval tau, issued at the start of interval t (load known up to t-1):
    profile of tau's day type (workday / not) over the 28 days before t's local date, at tau's
    local clock slot, plus the last residual decayed by phi**(h+1).
Quantiles add the empirical forecast-error quantile for that horizon, measured on the fitting
period (split-conformal style: errors are measured out of sample, on days the profile hadn't seen).
"""
from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from ..sim.scenario import ar1_noise

SLOTS = 96
LOOKBACK_DAYS = 28


def _local_slots(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = index.tz_convert(LOCAL_TZ)
    day = local.tz_localize(None).normalize()
    slot = np.asarray(local.hour * 4 + local.minute // 15)
    calendar = holidays.US(years=sorted(set(day.year)))
    work = np.array([d.weekday() < 5 and d not in calendar for d in day.date])
    return np.asarray(day.values), slot, work


@dataclass
class ProfileForecaster:
    days: np.ndarray        # local dates (datetime64) with a profile
    profile: np.ndarray     # [day, type(0 = non-workday, 1 = workday), slot]: as known at the start of that day
    phi: float
    horizons: int
    quantile_levels: tuple[float, ...]
    error_q: np.ndarray     # [horizon, quantile]

    @classmethod
    def fit(cls, load: pd.Series, fit_start: str, fit_end: str, horizons: int = 96,
            quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9, 0.99), stride: int = 4,
            noise_sigma: float = 0.0, noise_phi: float = 0.9, seed: int = 0) -> "ProfileForecaster":
        """Error quantiles are measured on the same series the controller will see: the real data, or a
        copy with the simulator's stress-test noise when noise_sigma > 0."""
        day, slot, work = _local_slots(load.index)
        frame = pd.DataFrame({"day": day, "slot": slot, "work": work, "kw": load.to_numpy()})
        grid = frame.pivot_table(index="day", columns="slot", values="kw", aggfunc="mean")
        grid = grid.reindex(columns=range(SLOTS)).interpolate(axis=1, limit_direction="both")
        day_work = frame.groupby("day")["work"].first().reindex(grid.index).to_numpy()
        values = grid.to_numpy()
        n_days = len(grid)
        profile = np.full((n_days, 2, SLOTS), np.nan)
        for d in range(n_days):
            lo = max(0, d - LOOKBACK_DAYS)
            for kind in (0, 1):
                rows = [r for r in range(lo, d) if day_work[r] == kind]
                if rows:
                    profile[d, kind] = values[rows].mean(axis=0)
        # Days at the very start have no history: fall back to the first available profile.
        for kind in (0, 1):
            first = next(d for d in range(n_days) if not np.isnan(profile[d, kind, 0]))
            profile[:first, kind] = profile[first, kind]
        model = cls(days=np.asarray(grid.index.values), profile=profile, phi=0.0, horizons=horizons,
                    quantile_levels=quantile_levels, error_q=np.zeros((horizons, len(quantile_levels))))

        inside = (load.index >= pd.Timestamp(fit_start, tz=LOCAL_TZ)) & (load.index < pd.Timestamp(fit_end, tz=LOCAL_TZ))
        noisy = load.to_numpy() * (1 + ar1_noise(len(load), noise_sigma, noise_phi, np.random.default_rng(seed)))
        base = model._base(load.index)
        resid = noisy - base
        r = resid[inside]
        model.phi = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0, 0.999))
        issues = np.flatnonzero(inside)[1::stride]
        issues = issues[issues + horizons < len(load)]
        errors = np.empty((len(issues), horizons))
        for h in range(horizons):
            point = base[issues + h] + resid[issues - 1] * model.phi ** (h + 1)
            errors[:, h] = noisy[issues + h] - point
        model.error_q = np.quantile(errors, quantile_levels, axis=0).T
        return model

    def _day_pos(self, days: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.days, days), 0, len(self.days) - 1)

    def _base(self, index: pd.DatetimeIndex, issue_days: np.ndarray | None = None) -> np.ndarray:
        day, slot, work = _local_slots(index)
        pos = self._day_pos(day if issue_days is None else issue_days)
        return self.profile[pos, work.astype(int), slot]

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray) -> np.ndarray:
        """Quantile forecasts [issue t, horizon h, quantile] for every interval of `index`.

        Row t uses only measured[:t]; targets past the end of `index` repeat the last interval.
        """
        n, horizons = len(index), self.horizons
        day, _, _ = _local_slots(index)
        targets = np.minimum(np.arange(n)[:, None] + np.arange(horizons)[None, :], n - 1)
        target_index = index[targets.ravel()]
        issue_days = np.repeat(day, horizons)
        base = self._base(target_index, issue_days).reshape(n, horizons)
        own = self._base(index)
        last_resid = np.concatenate([[0.0], measured[:-1] - own[:-1]])
        point = base + last_resid[:, None] * self.phi ** (np.arange(horizons) + 1)[None, :]
        return point[:, :, None] + self.error_q[None, :, :]
