"""Hosting-capacity Monte Carlo on real-session fleets.

    python scripts/capacity_study.py dev     # train/validation months (development)
    python scripts/capacity_study.py test    # summer 2017 test months (final numbers)

Writes data/results/capacity_runs_<split>.csv and capacity_summary_<split>.csv.
"""
from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from headroom.capacity import hosting_capacity, summarise  # noqa: E402
from headroom.config import DATA, PROCESSED  # noqa: E402
from headroom.control.baselines import LeastLaxityFirst, StaticCap  # noqa: E402
from headroom.control.headroom_mpc import HeadroomMPC  # noqa: E402
from headroom.forecast.precomputed import PrecomputedForecaster  # noqa: E402
from headroom.metrics import evaluate  # noqa: E402
from headroom.sim.engine import simulate  # noqa: E402
from headroom.sim.scenario import make_scenario  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402

SPLITS = {
    "dev": ["2016-07", "2016-08", "2017-03"],
    "test": ["2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10"],
}
PORTS = [int(x) for x in os.environ.get("CAPACITY_PORTS", "12,24,36,48,60,72").split(",")]
SEEDS = [int(x) for x in os.environ.get("CAPACITY_SEEDS", "1,2,3").split(",")]
CONTROLLERS = os.environ.get("CAPACITY_CONTROLLERS", "static_cap,llf,headroom").split(",")

_W: dict = {}


def _load() -> None:
    _W["site"] = pd.read_parquet(PROCESSED / "site_15min.parquet")
    _W["sessions"] = pd.read_parquet(PROCESSED / "sessions.parquet")
    _W["spec"] = load_gold("e-4-tou_2026")


def month_bounds(month: str) -> tuple[str, str]:
    first = pd.Timestamp(f"{month}-01")
    last = first + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    return str(first.date()), str(last.date())


def run_one(job: tuple) -> dict:
    month, n_ports, seed, name = job
    site, sessions, spec = _W["site"], _W["sessions"], _W["spec"]
    first, last = month_bounds(month)
    scenario = make_scenario(site, sessions, first, last, n_ports, seed=seed)
    cap = scenario.service_kw - site.loc[site["split"] == "train", "building_kw"].max()
    controller = {"static_cap": lambda: StaticCap(cap), "llf": lambda: LeastLaxityFirst(cap),
                  "headroom": lambda: HeadroomMPC(PrecomputedForecaster(), spec, site["building_kw"])}[name]()
    row = evaluate(simulate(scenario, controller), scenario, spec)
    row.update(month=month, seed=seed)
    return row


def main(split: str) -> None:
    jobs = list(itertools.product(SPLITS[split], PORTS, SEEDS, CONTROLLERS))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", "4")), initializer=_load) as pool:
        runs = pd.DataFrame(list(pool.map(run_one, jobs)))
    out = DATA / "results"
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / f"capacity_runs_{split}.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(out / f"capacity_summary_{split}.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print("hosting capacity (chargers):", hosting_capacity(summary).to_dict())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
