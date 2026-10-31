"""Final benchmark and ablations on the held-out test months (May-Oct 2017, summer demand charges).

    python scripts/final_benchmark.py            # 24 chargers (4x today's garage), seeds 1-3

Controllers: static cap, least-laxity-first, Caltech adacharge MPC (unmodified), Headroom (on the
hour-of-day profile forecast: no trained model at runtime).
Ablations: perfect-foresight oracle (upper bound), strict "never set a new peak" policy, the LightGBM
blend forecast instead of the profile (what a trained model would add), and a labelled stress test
with intra-hour variability (sigma 2%).
Writes data/results/final_runs.csv and final_summary.csv.
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

from headroom.config import DATA, PROCESSED  # noqa: E402
from headroom.control.baselines import AdachargeMPC, LeastLaxityFirst, StaticCap  # noqa: E402
from headroom.control.headroom_mpc import HeadroomMPC  # noqa: E402
from headroom.forecast.oracle import OracleForecaster  # noqa: E402
from headroom.forecast.precomputed import PrecomputedForecaster  # noqa: E402
from headroom.forecast.profile import ProfileForecaster  # noqa: E402
from headroom.metrics import evaluate, ramp_breaches  # noqa: E402
from headroom.sim.engine import SafetyLayer, simulate  # noqa: E402
from headroom.sim.scenario import make_scenario  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402
from headroom.thermal import TransformerModel  # noqa: E402

MONTHS = ["2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10"]
N_PORTS = int(os.environ.get("BENCH_PORTS", "24"))
SEEDS = [int(x) for x in os.environ.get("BENCH_SEEDS", "1,2,3").split(",")]
CONTROLLERS = os.environ.get("BENCH_CONTROLLERS", "static_cap,llf,adacharge_mpc,headroom,headroom_oracle,"
                             "headroom_strict,headroom_lightgbm,headroom_stress,llf_stress").split(",")

_W: dict = {}


def _load() -> None:
    _W["site"] = pd.read_parquet(PROCESSED / "site_15min.parquet")
    _W["sessions"] = pd.read_parquet(PROCESSED / "sessions.parquet")
    _W["spec"] = load_gold("e-4-tou_2026")


def run_one(job: tuple) -> dict:
    month, seed, name = job
    site, sessions, spec = _W["site"], _W["sessions"], _W["spec"]
    first = pd.Timestamp(f"{month}-01")
    last = first + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    stress = name.endswith("_stress")
    scenario = make_scenario(site, sessions, str(first.date()), str(last.date()), N_PORTS, seed=seed,
                             noise_sigma=0.02 if stress else 0.0)
    cap = scenario.service_kw - site.loc[site["split"] == "train", "building_kw"].max()
    history = site["building_kw"]
    safety = SafetyLayer(scenario.service_kw, ramp_kw=90.0 if stress else 60.0)
    if name.startswith("static_cap"):
        controller = StaticCap(cap)
    elif name.startswith("llf"):
        controller = LeastLaxityFirst(cap)
    elif name == "adacharge_mpc":
        controller = AdachargeMPC(cap)
    else:
        if name == "headroom_oracle":
            forecaster = OracleForecaster()
        elif name == "headroom_stress":
            # the stress run's meter differs from the saved forecasts' inputs, so it forecasts live
            forecaster = ProfileForecaster.fit(history, "2016-01-01", "2017-05-01", noise_sigma=0.02)
        else:
            forecaster = PrecomputedForecaster("lgbm" if name == "headroom_lightgbm" else "profile")
        controller = HeadroomMPC(forecaster, spec, history, unserved_usd_per_kwh=2.0 if name == "headroom_strict" else 25.0)
    result = simulate(scenario, controller, safety)
    row = evaluate(result, scenario, spec)
    w = scenario.n_window
    temp = site["temp_c"].reindex(scenario.index[:w]).to_numpy()
    row.update(controller=name, month=month, seed=seed, ramp_breaches=ramp_breaches(scenario, safety.ramp_kw),
               transformer_loss_of_life_h=TransformerModel().loss_of_life_hours(scenario.building_kw[:w] + result.ev_kw[:w], temp))
    return row


def main() -> None:
    jobs = list(itertools.product(MONTHS, SEEDS, CONTROLLERS))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", "4")), initializer=_load) as pool:
        runs = pd.DataFrame(list(pool.map(run_one, jobs)))
    out = DATA / "results"
    runs.to_csv(out / "final_runs.csv", index=False)
    cols = ["energy_served", "cars_short_over_1kwh", "added_demand_usd", "ev_energy_usd", "peak_site_kw",
            "service_violations", "safety_interventions", "ramp_breaches", "transformer_loss_of_life_h", "seconds"]
    summary = runs.groupby("controller")[cols].mean()
    summary["months"] = runs.groupby("controller").size()
    summary.to_csv(out / "final_summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
