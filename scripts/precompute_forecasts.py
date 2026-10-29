"""Walk-forward building-load forecasts, one file per month.

    python scripts/precompute_forecasts.py 2016-07 2016-08 2017-03 2017-05 ... 2017-12

For month M: LightGBM quantiles trained on everything before M - 2 months, conformally calibrated on
the two months before M, re-calibrated daily on the last 28 days (past data only), with the P99 tail
taken as the larger of LightGBM's and the hour-of-day profile's. Saves [issue, horizon, quantile]
for every interval of M plus one day, and the coverage of the month for the record.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(var, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import headroom.forecast.quantile_gbm as qg  # noqa: E402
from headroom.config import LOCAL_TZ, PROCESSED  # noqa: E402
from headroom.forecast.precomputed import FORECASTS  # noqa: E402
from headroom.forecast.profile import ProfileForecaster  # noqa: E402

qg.PARAMS["n_jobs"] = 4


def month_window(month: str) -> pd.DatetimeIndex:
    site = pd.read_parquet(PROCESSED / "site_15min.parquet")
    start = pd.Timestamp(f"{month}-01", tz=LOCAL_TZ)
    stop = start + pd.DateOffset(months=1) + pd.Timedelta(days=1)
    return site.index[(site.index >= start) & (site.index < stop)]


def main(months: list[str]) -> None:
    site = pd.read_parquet(PROCESSED / "site_15min.parquet")
    load, temp = site["building_kw"], site["temp_c"]
    FORECASTS.mkdir(parents=True, exist_ok=True)
    for month in months:
        started = time.time()
        gbm = qg.QuantileGBM.fit(load, temp, train_end=f"{month}-01", stride=8)
        profile = ProfileForecaster.fit(load, "2016-01-01", f"{month}-01")
        model = qg.TailEnsemble(qg.RollingConformal(gbm), profile)
        index = month_window(month)
        pred = model.predict(index, load.loc[index].to_numpy()).astype(np.float32)
        np.savez_compressed(FORECASTS / f"{month}.npz", q=pred, start=str(index[0]), n=len(index),
                            levels=np.array(model.quantile_levels))
        last_day = str((pd.Timestamp(f"{month}-01") + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date())
        cov = qg.coverage(model, load, f"{month}-01", last_day)
        (FORECASTS / f"{month}.json").write_text(cov.to_json(orient="records"), encoding="utf-8")
        print(month, f"{time.time() - started:.0f}s", json.dumps(cov.round(4).to_dict("records")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
