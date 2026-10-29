"""Development sweep on train/validation months only (the 2017-05..12 test split stays untouched).

    python scripts/dev_sweep.py

Writes data/results/dev_sweep.csv: every controller x port count x month.
"""
from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")  # one math thread per worker: 8 GB RAM, several workers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headroom.config import DATA, PROCESSED  # noqa: E402
from headroom.control.baselines import LeastLaxityFirst  # noqa: E402
from headroom.control.headroom_mpc import HeadroomMPC  # noqa: E402
from headroom.forecast.oracle import OracleForecaster  # noqa: E402
from headroom.forecast.profile import ProfileForecaster  # noqa: E402
from headroom.metrics import evaluate  # noqa: E402
from headroom.sim.engine import simulate  # noqa: E402
from headroom.sim.scenario import make_scenario  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402

MONTHS = [("2016-07-01", "2016-07-31"), ("2016-08-01", "2016-08-31"), ("2016-09-01", "2016-09-30"),
          ("2017-02-01", "2017-02-28"), ("2017-03-01", "2017-03-31")]
PORTS = [6, 12, 18, 24, 36]
CONTROLLERS = ["llf", "headroom", "headroom_q90", "headroom_oracle"]


_W: dict = {}


def _load() -> None:
    _W["site"] = pd.read_parquet(PROCESSED / "site_15min.parquet")
    _W["sessions"] = pd.read_parquet(PROCESSED / "sessions.parquet")
    _W["spec"] = load_gold("e-4-tou_2026")
    _W["profile"] = ProfileForecaster.fit(_W["site"]["building_kw"], "2016-01-01", "2017-05-01")


def run_one(job: tuple) -> dict:
    (first, last), n_ports, name = job
    site, sessions, spec = _W["site"], _W["sessions"], _W["spec"]
    history = site["building_kw"]
    scenario = make_scenario(site, sessions, first, last, n_ports, seed=1)
    cap = scenario.service_kw - site.loc[site["split"] == "train", "building_kw"].max()
    if name == "llf":
        controller = LeastLaxityFirst(cap)
    else:
        forecaster = OracleForecaster() if name == "headroom_oracle" else _W["profile"]
        controller = HeadroomMPC(forecaster, spec, history, q_high=0.9 if name == "headroom_q90" else 0.99)
        controller.name = name
    row = evaluate(simulate(scenario, controller), scenario, spec)
    row["month"] = first[:7]
    return row


def main() -> None:
    jobs = list(itertools.product(MONTHS, PORTS, CONTROLLERS))
    with ProcessPoolExecutor(max_workers=4, initializer=_load) as pool:
        rows = list(pool.map(run_one, jobs))
    out = DATA / "results"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "dev_sweep.csv", index=False)
    summary = frame.groupby(["controller", "n_ports"])[
        ["energy_served", "cars_short_over_1kwh", "new_demand_peaks", "added_demand_usd"]].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
