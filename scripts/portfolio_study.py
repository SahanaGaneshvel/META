"""Headroom Check across many real buildings: does the gain hold beyond our one site?

    python scripts/portfolio_study.py select     # pick buildings, save their meters to data/processed
    python scripts/portfolio_study.py run        # screen each one (hours on a laptop; 4 workers)
    PORTFOLIO_SHARE=0.9 python scripts/portfolio_study.py run   # sensitivity: tighter service size

Buildings: BDG2 electricity meters at US sites, 2016-2017, kept only if the meter is near-complete
(<= 2% missing hours, no zero/flat stretch longer than a day) and the peak is 100-2,500 kW (sites that
pay demand charges and could host a garage). A fixed-seed sample, spread across sites and building
uses. Each building is screened on the held-out summer months (Jul-Aug 2017) with its service size
from the 80% continuous-load rule (it isn't in the data), real Palo Alto sessions and the E-4 TOU tariff.
Writes data/results/portfolio_runs.csv (every simulated month) and portfolio_summary.csv.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from headroom.check import check, site_table  # noqa: E402
from headroom.config import DATA, PROCESSED, RAW  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402

N_BUILDINGS = int(os.environ.get("PORTFOLIO_N", "50"))
SHARE = float(os.environ.get("PORTFOLIO_SHARE", "0.8"))  # service = peak before the test months / SHARE
MONTHS = ["2017-07", "2017-08"]
METERS = PROCESSED / "portfolio_meters.parquet"
OUT = DATA / "results"
TAG = "" if SHARE == 0.8 else f"_share{int(SHARE * 100)}"
PARTS = OUT / f"portfolio_parts{TAG}"
_W: dict = {}


def select() -> None:
    meta = pd.read_csv(RAW / "bdg2" / "metadata.csv")
    us = meta[meta["timezone"].str.startswith("US/") & (meta["electricity"] == "Yes")]
    raw = pd.read_csv(RAW / "bdg2" / "electricity_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp")
    raw = raw[[c for c in us["building_id"] if c in raw.columns]]
    keep = []
    for b in raw.columns:
        s = raw[b]
        if s.isna().mean() > 0.02 or not 100 <= s.max() <= 2500:
            continue
        flat = (s.diff().fillna(1) == 0).astype(int)
        longest_flat = flat.groupby((flat == 0).cumsum()).sum().max()
        if longest_flat > 24 or (s <= 0).mean() > 0.01:
            continue
        keep.append(b)
    pool = us.set_index("building_id").loc[keep]
    rng = np.random.default_rng(2026)
    # spread across sites: round-robin over sites, random order within each
    by_site = {site: list(rng.permutation(g.index)) for site, g in pool.groupby("site_id")}
    picked = []
    while len(picked) < min(N_BUILDINGS, len(pool)):
        for site in sorted(by_site):
            if by_site[site] and len(picked) < N_BUILDINGS:
                picked.append(by_site[site].pop())
    raw[picked].to_parquet(METERS)
    info = pool.loc[picked, ["site_id", "primaryspaceusage", "sqm", "timezone"]]
    info.to_csv(OUT / "portfolio_buildings.csv")
    print(f"{len(keep)} buildings pass the quality filter; picked {len(picked)} across "
          f"{info['site_id'].nunique()} sites:\n{info['primaryspaceusage'].value_counts().to_string()}")


def _load() -> None:
    _W["meters"] = pd.read_parquet(METERS)
    _W["sessions"] = pd.read_parquet(PROCESSED / "sessions.parquet")
    _W["spec"] = load_gold("e-4-tou_2026")


def screen(building: str) -> dict:
    started = time.time()
    meter = _W["meters"][building].interpolate(limit=3).dropna()
    meter.attrs["interval"] = "1h"
    site_df = site_table(meter)
    out = check(site_df, _W["sessions"], _W["spec"], MONTHS, continuous_share=SHARE)
    runs = out.pop("runs")
    runs["building"] = building
    out.update(building=building, seconds=round(time.time() - started))
    runs.to_csv(PARTS / f"{building}.csv", index=False)
    pd.Series({k: v for k, v in out.items() if not isinstance(v, (dict, list))}).to_json(PARTS / f"{building}.json")
    print(building, out["chargers_standard_smart_charging"], "->", out["chargers_headroom"], f"{out['seconds']}s", flush=True)
    return out


def run() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    buildings = list(pd.read_parquet(METERS).columns)
    done = {p.stem for p in PARTS.glob("*.json")}
    todo = [b for b in buildings if b not in done]  # resumable
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", "4")), initializer=_load) as pool:
        for _ in pool.map(screen, todo):
            pass
    summarise()


def summarise() -> None:
    runs = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob("*.csv"))], ignore_index=True)
    runs.to_csv(OUT / f"portfolio_runs{TAG}.csv", index=False)
    s = pd.DataFrame([pd.read_json(p, typ="series") for p in sorted(PARTS.glob("*.json"))])
    for col in ("chargers_standard_smart_charging", "chargers_headroom", "extra_chargers", "service_violations"):
        s[col] = s[col].astype(int)
    s = s.merge(pd.read_csv(OUT / "portfolio_buildings.csv"), left_on="building", right_on="building_id")
    s.drop(columns="building_id").to_csv(OUT / f"portfolio_summary{TAG}.csv", index=False)
    std, head = s["chargers_standard_smart_charging"], s["chargers_headroom"]
    both = s[std > 0]
    gain = (both["chargers_headroom"] / both["chargers_standard_smart_charging"] - 1).to_numpy()
    rng = np.random.default_rng(0)
    boot = np.median(rng.choice(gain, (10_000, len(gain))), axis=1)
    print(f"service = peak before the test months / {SHARE}; {len(s)} buildings, {s['site_id'].nunique()} sites, "
          f"{len(runs)} simulated building-months; service-limit violations: {int(s['service_violations'].sum())}")
    print(f"chargers (median): standard smart charging {std.median():.0f}, Headroom {head.median():.0f}; "
          f"total {std.sum()} -> {head.sum()} ({head.sum() / max(std.sum(), 1) - 1:+.0%})")
    print(f"per-building gain where standard > 0 (n={len(both)}): median {np.median(gain):+.0%} "
          f"(95% bootstrap CI {np.quantile(boot, .025):+.0%}..{np.quantile(boot, .975):+.0%}), "
          f"IQR {np.quantile(gain, .25):+.0%}..{np.quantile(gain, .75):+.0%}; more chargers in "
          f"{(both['extra_chargers'] > 0).mean():.0%}, fewer in {(both['extra_chargers'] < 0).mean():.0%}; "
          f"standard fits none in {(std == 0).sum()}, Headroom fits none in {(head == 0).sum()}")


if __name__ == "__main__":
    {"select": select, "run": run, "summarise": summarise}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
