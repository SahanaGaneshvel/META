"""Site definition and project paths. Rationale for every value: docs/DATA_SPEC.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

LOCAL_TZ = "America/Los_Angeles"
INTERVAL = pd.Timedelta("15min")


@dataclass(frozen=True)
class SiteConfig:
    """The virtual Bay Area site: Palo Alto garage sessions + a Berkeley building on CPAU E-4 TOU."""

    garage: str = "WEBSTER"                    # Palo Alto garage, 3 stations x 2 ports
    building: str = "Bear_education_Patricia"  # BDG2 meter, UC Berkeley
    weather_site: str = "Bear"
    weather_latlon: tuple[float, float] = (37.8719, -122.2585)
    port_kw_cap: float = 6.2                   # measured p99 charging power
    service_limit_kw: float = 1000.0           # assumed 1,000 kVA transformer, pf ~ 1
    first_day: str = "2016-01-01"
    end_day: str = "2018-01-01"                # exclusive
    splits: dict[str, tuple[str, str]] = field(default_factory=lambda: {
        "train": ("2016-01-01", "2017-01-01"),
        "val": ("2017-01-01", "2017-05-01"),
        "test": ("2017-05-01", "2018-01-01"),
    })
    # CAISO statewide Flex Alerts: local date -> (start hour, end hour); None = hours not published.
    flex_alerts: dict[str, tuple[int, int] | None] = field(default_factory=lambda: {
        "2016-06-20": None,
        "2016-07-27": None,
        "2016-07-28": None,
        "2017-06-20": (14, 21),
        "2017-06-21": (14, 21),
        "2017-09-01": (13, 22),
    })

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.first_day, tz=LOCAL_TZ)

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.end_day, tz=LOCAL_TZ)

    def grid(self, freq: str | pd.Timedelta = INTERVAL) -> pd.DatetimeIndex:
        """UTC timestamps for every interval start in [start, end); uniform across DST changes."""
        return pd.date_range(self.start.tz_convert("UTC"), self.end.tz_convert("UTC"), freq=freq,
                             inclusive="left", name="ts_utc")

    def split_labels(self, local_days: pd.DatetimeIndex) -> pd.Series:
        days = pd.DatetimeIndex(local_days).tz_localize(None).normalize()
        labels = pd.Series("", index=range(len(days)), dtype=object)
        for name, (lo, hi) in self.splits.items():
            labels[(days >= pd.Timestamp(lo)) & (days < pd.Timestamp(hi))] = name
        return labels


SITE = SiteConfig()
