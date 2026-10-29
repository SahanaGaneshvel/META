"""Distribution-transformer hot-spot temperature and insulation ageing (IEEE C57.91-2011, clause 7).

Exponential top-oil and winding responses, driven by per-unit load K and ambient temperature, then
the ageing acceleration factor F_AA = exp(15000/383 - 15000/(theta_H + 273)) (1.0 at a 110 C hot
spot). Parameters are the standard's typical values for an oil-immersed, self-cooled (ONAN) 65 C
rise unit; the rating is the site's service limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransformerModel:
    rated_kva: float = 1000.0
    power_factor: float = 1.0
    top_oil_rise_c: float = 55.0      # rated top-oil rise over ambient
    hot_spot_rise_c: float = 25.0     # rated hot-spot rise over top oil
    loss_ratio: float = 4.5           # R: load loss / no-load loss at rated load
    n: float = 0.8                    # oil exponent (ONAN)
    m: float = 0.8                    # winding exponent (ONAN)
    tau_oil_min: float = 180.0
    tau_winding_min: float = 7.0

    def hot_spot(self, load_kw: np.ndarray, ambient_c: np.ndarray, step_min: float = 15.0) -> np.ndarray:
        k = np.asarray(load_kw) / (self.rated_kva * self.power_factor)
        oil_target = self.top_oil_rise_c * ((k**2 * self.loss_ratio + 1) / (self.loss_ratio + 1)) ** self.n
        wind_target = self.hot_spot_rise_c * k ** (2 * self.m)
        a_oil = np.exp(-step_min / self.tau_oil_min)
        a_wind = np.exp(-step_min / self.tau_winding_min)
        oil = np.empty_like(oil_target)
        wind = np.empty_like(wind_target)
        oil[0], wind[0] = oil_target[0], wind_target[0]
        for i in range(1, len(k)):
            oil[i] = oil_target[i] + (oil[i - 1] - oil_target[i]) * a_oil
            wind[i] = wind_target[i] + (wind[i - 1] - wind_target[i]) * a_wind
        return np.asarray(ambient_c) + oil + wind

    def loss_of_life_hours(self, load_kw: np.ndarray, ambient_c: np.ndarray, step_min: float = 15.0) -> float:
        theta = self.hot_spot(load_kw, ambient_c, step_min)
        faa = np.exp(15000 / 383 - 15000 / (theta + 273))
        return float(faa.sum() * step_min / 60)
