"""Perfect foresight: every quantile is the true future load. An upper bound for ablations, never deployable."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OracleForecaster:
    horizons: int = 96
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9, 0.99)

    def predict(self, index: pd.DatetimeIndex, measured: np.ndarray) -> np.ndarray:
        n = len(index)
        targets = np.minimum(np.arange(n)[:, None] + np.arange(self.horizons)[None, :], n - 1)
        truth = measured[targets]
        return np.repeat(truth[:, :, None], len(self.quantile_levels), axis=2)
