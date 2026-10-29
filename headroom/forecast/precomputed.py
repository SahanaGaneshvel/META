"""Read walk-forward forecasts saved by scripts/precompute_forecasts.py."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DATA

FORECASTS = DATA / "forecasts"


@dataclass
class PrecomputedForecaster:
    horizons: int = 96
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9, 0.99)

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray) -> np.ndarray:
        """Forecasts for `index`, which must start at a month boundary covered by a saved file."""
        out = np.empty((len(index), self.horizons, len(self.quantile_levels)))
        filled = 0
        while filled < len(index):
            month = index[filled].tz_convert("America/Los_Angeles").strftime("%Y-%m")
            path = FORECASTS / f"{month}.npz"
            if not path.exists():
                raise FileNotFoundError(f"no forecasts for {month}: run scripts/precompute_forecasts.py {month}")
            saved = np.load(path)
            start = pd.Timestamp(str(saved["start"]))
            offset = int((index[filled] - start) / pd.Timedelta("15min"))
            take = min(len(index) - filled, int(saved["n"]) - offset)
            out[filled:filled + take] = saved["q"][offset:offset + take]
            filled += take
        return out


@dataclass
class ArrayForecaster:
    """Serves a forecast array already aligned with a scenario (e.g. after an accepted adjustment)."""
    array: np.ndarray
    horizons: int = 96
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9, 0.99)

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray) -> np.ndarray:
        return self.array[: len(index)]
