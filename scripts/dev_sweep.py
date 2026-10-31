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
from headroom.control.baselines import LeastLaxityFirst, StaticCap  # noqa: E402
from headroom.control.headroom_mpc import HeadroomMPC  # noqa: E402
from headroom.forecast.oracle import OracleForecaster  # noqa: E402
from headroom.forecast.profile import ProfileForecaster  # noqa: E402
from headroom.metrics import evaluate  # noqa: E402
from headroom.sim.engine import simulate  # noqa: E402
from headroom.sim.scenario import make_scenario  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402

MONTHS = [("2016-07-01", "2016-07-31"), ("2016-08-01", "2016-08-31"), ("2017-03-01", "2017-03-31")]
PORTS = [int(x) for x in os.environ.get("SWEEP_PORTS", "12,24,48,72").split(",")]
CONTROLLERS = os.environ.get("SWEEP_CONTROLLERS", "static_cap,llf,headroom,headroom_strict").split(",")
OUT_NAME = os.environ.get("SWEEP_OUT", "dev_sweep")


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
    elif name == "static_cap":
        controller = StaticCap(cap)
    else:
        forecaster = OracleForecaster() if name.endswith("oracle") else _W["profile"]
        controller = HeadroomMPC(forecaster, spec, history,
                                 unserved_usd_per_kwh=2.0 if "strict" in name else None,
                                 oracle_floors="full_oracle" in name)
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
    frame.to_csv(out / f"{OUT_NAME}.csv", index=False)
    summary = frame.groupby(["controller", "n_ports"])[
        ["energy_served", "cars_short_over_1kwh", "new_demand_peaks", "added_demand_usd",
         "service_violations", "safety_interventions", "peak_site_kw"]].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
