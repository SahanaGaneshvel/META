"""LightGBM quantile forecaster on top of the hour-of-day profile, with split-conformal calibration.

Direct multi-horizon: one row per (issue time t, horizon h), horizon as a feature. The target is the
residual of the building load from the profile forecast, so the trees learn what the profile
misses (weather, level shifts, recent dynamics). Every feature for row (t, h) uses data measured
before t; no future weather is used (docs/DATA_SPEC.md, section 3G).

Calibration (one-sided split conformal per quantile and horizon bucket): on the calibration months,
each quantile is shifted by the empirical quantile of its own errors, so a P90 really is exceeded
10% of the time on recent data. Walk-forward: `walk_forward` refits at the start of every month on
all earlier data and calibrates on the two months before it, the way it would run in deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .profile import ProfileForecaster, _local_slots

LEVELS = (0.1, 0.5, 0.9, 0.99)
BUCKETS = (0, 4, 16, 48, 96)  # horizon buckets for calibration: <1 h, 1-4 h, 4-12 h, 12-24 h
FEATURES = ["h", "slot", "dow", "work", "month", "base", "last", "resid", "mean4", "slope4",
            "lag_day", "lag_week", "temp_last", "temp_day_max"]
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)


def _features(load: np.ndarray, temp: np.ndarray, index: pd.DatetimeIndex, profile: ProfileForecaster,
              issues: np.ndarray, horizons: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature rows for every (issue, horizon); returns (X, target positions)."""
    n_issue = len(issues)
    h = np.tile(np.arange(horizons), n_issue)
    t = np.repeat(issues, horizons)
    target = np.minimum(t + h, len(load) - 1)
    day, slot, work = _local_slots(index)
    local = index.tz_convert(LOCAL_TZ)
    base_target = profile._base(index[target], issue_days=day[t])
    own_base = profile._base(index)
    last = load[t - 1]
    resid = last - own_base[t - 1]
    window = np.stack([load[np.maximum(t - k, 0)] for k in range(1, 5)])
    lag_day_pos = np.where(target - 96 < t, target - 96, target - 192)
    temp_day_max = pd.Series(temp).rolling(96, min_periods=1).max().to_numpy()
    X = pd.DataFrame({
        "h": h, "slot": slot[target], "dow": np.asarray(local.day_of_week)[target], "work": work[target].astype(int),
        "month": np.asarray(local.month)[target], "base": base_target, "last": last, "resid": resid,
        "mean4": window.mean(axis=0), "slope4": window[0] - window[-1],
        "lag_day": load[np.maximum(lag_day_pos, 0)], "lag_week": load[np.maximum(target - 672, 0)],
        "temp_last": temp[t - 1], "temp_day_max": temp_day_max[t - 1],
    })
    return X, target


@dataclass
class QuantileGBM:
    profile: ProfileForecaster
    models: dict = field(default_factory=dict)
    shift: np.ndarray | None = None        # [bucket, level] conformal corrections
    horizons: int = 96
    quantile_levels: tuple[float, ...] = LEVELS
    history_load: pd.Series | None = None
    history_temp: pd.Series | None = None

    @classmethod
    def fit(cls, load: pd.Series, temp: pd.Series, train_end: str, calib_months: int = 2,
            stride: int = 4, seed: int = 0) -> "QuantileGBM":
        """Train on data before (train_end - calib_months); calibrate on the calib months."""
        end = pd.Timestamp(train_end, tz=LOCAL_TZ)
        calib_start = end - pd.DateOffset(months=calib_months)
        profile = ProfileForecaster.fit(load, str(load.index[0].tz_convert(LOCAL_TZ).date()), str(calib_start.date()))
        model = cls(profile=profile, history_load=load, history_temp=temp)
        values, temps, index = load.to_numpy(), temp.to_numpy(), load.index
        first = 672 + 1  # need a week of lags
        train_issues = np.arange(first, np.searchsorted(index, calib_start) - 96, stride)
        X, target = _features(values, temps, index, profile, train_issues, model.horizons)
        y = values[target] - X["base"].to_numpy()
        for level in model.quantile_levels:
            m = lgb.LGBMRegressor(objective="quantile", alpha=level, random_state=seed, **PARAMS)
            model.models[level] = m.fit(X[FEATURES], y)

        calib_issues = np.arange(np.searchsorted(index, calib_start), np.searchsorted(index, end) - 96, stride)
        Xc, target_c = _features(values, temps, index, profile, calib_issues, model.horizons)
        yc = values[target_c] - Xc["base"].to_numpy()
        model.shift = np.zeros((len(BUCKETS) - 1, len(model.quantile_levels)))
        for j, level in enumerate(model.quantile_levels):
            err = yc - model.models[level].predict(Xc[FEATURES])
            for b in range(len(BUCKETS) - 1):
                sel = (Xc["h"] >= BUCKETS[b]) & (Xc["h"] < BUCKETS[b + 1])
                model.shift[b, j] = np.quantile(err[sel], level)
        return model

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray) -> np.ndarray:
        """[issue t, horizon h, quantile] for each interval of `index` (same contract as ProfileForecaster)."""
        full_load = self.history_load.copy()
        pos = full_load.index.get_indexer(index)
        full_load.iloc[pos] = measured  # the controller's own meter readings inside the window
        values, temps, full_index = full_load.to_numpy(), self.history_temp.to_numpy(), full_load.index
        X, _ = _features(values, temps, full_index, self.profile, pos, self.horizons)
        bucket = np.searchsorted(BUCKETS, X["h"].to_numpy(), side="right") - 1
        out = np.empty((len(index), self.horizons, len(self.quantile_levels)))
        base = X["base"].to_numpy()
        for j, level in enumerate(self.quantile_levels):
            q = base + self.models[level].predict(X[FEATURES]) + self.shift[bucket, j]
            out[:, :, j] = q.reshape(len(index), self.horizons)
        out.sort(axis=2)  # keep quantiles from crossing
        return out


def coverage(forecaster, load: pd.Series, first: str, last: str, stride: int = 8) -> pd.DataFrame:
    """Empirical coverage and pinball loss per quantile over [first, last], issued every `stride` steps."""
    start = pd.Timestamp(first, tz=LOCAL_TZ)
    stop = pd.Timestamp(last, tz=LOCAL_TZ) + pd.Timedelta(days=1)
    window = load[(load.index >= start) & (load.index < stop + pd.Timedelta(days=1))]
    pred = forecaster.predict(window.index, window.to_numpy())
    n = int(((window.index >= start) & (window.index < stop)).sum())
    issues = np.arange(1, n, stride)
    targets = np.minimum(issues[:, None] + np.arange(forecaster.horizons)[None, :], len(window) - 1)
    truth = window.to_numpy()[targets]
    rows = []
    for j, level in enumerate(forecaster.quantile_levels):
        q = pred[issues, :, j]
        diff = truth - q
        rows.append({"level": level, "coverage": float((truth <= q).mean()),
                     "pinball": float(np.mean(np.maximum(level * diff, (level - 1) * diff)))})
    return pd.DataFrame(rows)


@dataclass
class RollingConformal:
    """Re-calibrates any quantile forecaster every day on its own recent errors (past data only).

    For issues on local day D, each quantile and horizon bucket is shifted by the empirical
    quantile of the base forecaster's errors over days D-window..D-2, whose targets were all
    measured before D. This tracks drift (season, occupancy) that a fixed calibration set misses.
    """
    base: object
    window_days: int = 28

    @property
    def horizons(self) -> int:
        return self.base.horizons

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        return self.base.quantile_levels

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray, history: pd.Series | None = None) -> np.ndarray:
        history = history if history is not None else self.base.history_load
        lead = pd.Timedelta(days=self.window_days + 1)
        full = history[(history.index >= index[0] - lead) & (history.index <= index[-1])].copy()
        full.loc[index] = measured
        raw = self.base.predict(full.index, full.to_numpy())
        n, horizons = len(full), self.horizons
        targets = np.minimum(np.arange(n)[:, None] + np.arange(horizons)[None, :], n - 1)
        errors = full.to_numpy()[targets][:, :, None] - raw          # [issue, h, level]
        day = np.asarray(full.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize().values)
        days = np.unique(day)
        bucket = np.searchsorted(BUCKETS, np.arange(horizons), side="right") - 1
        out = raw.copy()
        for d in days:
            rows = day == d
            past = (day >= d - np.timedelta64(self.window_days, "D")) & (day <= d - np.timedelta64(2, "D"))
            if past.sum() < 96 * 7:
                continue  # not enough history yet: keep the base calibration
            for j, level in enumerate(self.quantile_levels):
                for b in range(len(BUCKETS) - 1):
                    cols = bucket == b
                    out[np.ix_(rows, cols, [j])] += np.quantile(errors[np.ix_(past, cols, [j])], level)
        out.sort(axis=2)
        return out[len(full) - len(index):]


@dataclass
class TailEnsemble:
    """Sharp body, conservative tail: quantiles from `body`, except levels >= tail_from, which take
    the larger of `body` and `tail` (two independent models must both be wrong to under-cover)."""
    body: object
    tail: object
    tail_from: float = 0.99

    @property
    def horizons(self) -> int:
        return self.body.horizons

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        return self.body.quantile_levels

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray) -> np.ndarray:
        out = self.body.predict(index, measured)
        other = self.tail.predict(index, measured)
        for j, level in enumerate(self.quantile_levels):
            if level >= self.tail_from:
                out[:, :, j] = np.maximum(out[:, :, j], other[:, :, j])
        return out


@dataclass
class HorizonBlend:
    """Per-horizon model selection, chosen on validation months (Mar-Apr 2017, pinball loss):
    the persistence-corrected profile wins below 4 hours (it knows where the load just was),
    LightGBM wins from 4 to 24 hours. Beyond the switch, the P99 tail is the larger of the two."""
    short: object
    long: object
    switch: int = 16  # 15-minute steps (4 hours)
    tail_from: float = 0.99

    @property
    def horizons(self) -> int:
        return self.long.horizons

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        return self.long.quantile_levels

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray) -> np.ndarray:
        out = self.long.predict(index, measured)
        near = self.short.predict(index, measured)
        out[:, : self.switch] = near[:, : self.switch]
        for j, level in enumerate(self.quantile_levels):
            if level >= self.tail_from:
                out[:, self.switch:, j] = np.maximum(out[:, self.switch:, j], near[:, self.switch:, j])
        out.sort(axis=2)
        return out
