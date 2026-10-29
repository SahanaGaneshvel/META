"""The site the agent operates in the demo: a replay of a real month, answered from simulation.

Every tool answer is computed: from the simulation of the month under Headroom (and the baselines
for comparison), the walk-forward forecasts, the capacity study and the tariff verifiers. The model
never supplies a number itself.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..capacity import expansion_answer
from ..config import DATA, LOCAL_TZ, PROCESSED, SITE
from ..control.baselines import LeastLaxityFirst, StaticCap
from ..control.headroom_mpc import HeadroomMPC
from ..forecast.precomputed import PrecomputedForecaster
from ..metrics import evaluate
from ..sim.engine import Result, simulate
from ..sim.scenario import STEP_H, Scenario, make_scenario
from ..tariff import TariffSpec, interval_labels, load_gold
from .tools import (ExplainPlan, GridAlerts, GridEventAdjustment, ReadTariff, SetRequirement,
                    SimulateExpansion)

RESULTS = DATA / "results"
DOCS = DATA / "docs"
TARIFF_DOCS = {"cpau-e-4-tou-2026": ("e-4-tou_effective_2026-07-01.pdf", "e-4-tou_2026"),
               "cpau-e-7-tou-2026": ("e-7-tou_effective_2026-07-01.pdf", "e-7-tou_2026")}
LEVELS = (0.1, 0.5, 0.9, 0.99)


def _local(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return (t.tz_localize(LOCAL_TZ) if t.tzinfo is None else t).tz_convert("UTC")


@dataclass
class ReplaySite:
    month: str = "2017-06"
    n_ports: int = 24
    seed: int = 1
    now_local: str = "2017-06-21T15:00"
    spec_name: str = "e-4-tou_2026"
    requirements: list[dict] = field(default_factory=list)
    adjustments: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.site = pd.read_parquet(PROCESSED / "site_15min.parquet")
        self.sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
        self.spec: TariffSpec = load_gold(self.spec_name)
        first = pd.Timestamp(f"{self.month}-01")
        last = first + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        self.scenario: Scenario = make_scenario(self.site, self.sessions, str(first.date()), str(last.date()),
                                                self.n_ports, seed=self.seed)
        self.forecast = PrecomputedForecaster().predict(self.scenario.index, self.scenario.building_kw)
        self.results = self._simulate_all()
        self.labels = interval_labels(self.scenario.index, self.spec)

    # -- simulation, cached because each month takes about a minute ---------------------------
    def _simulate_all(self) -> dict[str, Result]:
        path = RESULTS / f"replay_{self.month}_{self.n_ports}_{self.seed}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        cap = SITE.service_limit_kw - self.site.loc[self.site["split"] == "train", "building_kw"].max()
        controllers = [HeadroomMPC(PrecomputedForecaster(), self.spec, self.site["building_kw"]),
                       LeastLaxityFirst(cap), StaticCap(cap)]
        results = {c.name: simulate(self.scenario, c) for c in controllers}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(results))
        return results

    def step_of(self, local_time: str) -> int:
        return int(np.clip(self.scenario.index.get_indexer([_local(local_time)], method="nearest")[0],
                           0, len(self.scenario.index) - 1))

    @property
    def t(self) -> int:
        return self.step_of(self.now_local)

    def _month_to_date(self, controller: str, t: int) -> dict:
        idx = self.scenario.index
        site_kw = self.scenario.building_kw + self.results[controller].ev_kw
        local = idx.tz_convert(LOCAL_TZ)
        same = (local.month == local[t].month) & (np.arange(len(idx)) < t)
        peak_window = self.labels.filter(like="demand:peak").any(axis=1).to_numpy()
        return {"max_kw": float(site_kw[same].max()) if same.any() else 0.0,
                "peak_window_kw": float(site_kw[same & peak_window].max()) if (same & peak_window).any() else 0.0}

    # -- tools ----------------------------------------------------------------------------------
    def site_status(self) -> dict:
        t = self.t
        sc, r = self.scenario, self.results["headroom"]
        plugged = int(((sc.arrive <= t) & (sc.depart > t)).sum())
        q = self.forecast[t]
        hours = [{"in_hours": h // 4, "building_kw_p50": round(float(q[h, 1])), "building_kw_p99": round(float(q[h, 3]))}
                 for h in range(0, 24, 4)]
        mtd = self._month_to_date("headroom", t)
        return {
            "local_time": self.now_local, "chargers": sc.n_ports, "cars_plugged_in": plugged,
            "building_kw_last_15min": round(float(sc.building_kw[t - 1])), "ev_kw_now": round(float(r.ev_kw[t]), 1),
            "service_limit_kw": sc.service_kw,
            "demand_already_paid_this_month_kw": {"any_time": round(mtd["max_kw"]), "4_to_9pm_weekdays": round(mtd["peak_window_kw"])},
            "building_forecast_next_hours": hours,
            "active_requirements": self.requirements, "accepted_grid_adjustments": self.adjustments,
        }

    def explain_plan(self, args: ExplainPlan) -> dict:
        t = self.step_of(args.local_time)
        sc = self.scenario
        r = self.results["headroom"]
        q = self.forecast[t, 0]
        mtd = self._month_to_date("headroom", t)
        peak = bool(self.labels.filter(like="demand:peak").any(axis=1).iloc[t])
        ceiling = mtd["peak_window_kw"] if peak else mtd["max_kw"]
        return {
            "local_time": args.local_time, "tariff_period": self.labels["period"].iloc[t],
            "in_4_to_9pm_demand_window": peak, "cars_plugged_in": int(((sc.arrive <= t) & (sc.depart > t)).sum()),
            "building_forecast_kw": {"p50": round(float(q[1])), "p99": round(float(q[3]))},
            "building_actual_kw": round(float(sc.building_kw[t])),
            "demand_level_already_paid_kw": round(ceiling),
            "ev_kw_headroom": round(float(r.ev_kw[t]), 1),
            "ev_kw_standard_smart_charger": round(float(self.results["llf"].ev_kw[t]), 1),
            "safety_layer_trimmed_kw": round(float(r.safety_clipped_kw[t]), 1),
            "site_kw_headroom": round(float(sc.building_kw[t] + r.ev_kw[t])),
        }

    def set_requirement(self, args: SetRequirement) -> dict:
        start, stop = self.step_of(args.plug_in_local), self.step_of(args.ready_by_local)
        if stop <= start:
            return {"accepted": False, "reason": "deadline is not after plug-in"}
        need = args.vehicles * args.kwh_per_vehicle
        q = self.forecast[start, : min(stop - start, self.forecast.shape[1])]
        steps = len(q)
        mtd = self._month_to_date("headroom", start)
        ceiling = max(mtd["max_kw"], float(q[:, 0].max()))
        ev = self.results["headroom"].ev_kw[start:start + steps]
        per_step_max = args.vehicles * SITE.port_kw_cap
        deliverable = {}
        for j, level in enumerate(LEVELS[1:], start=1):
            room = np.clip(ceiling - q[:, j] - ev, 0, per_step_max)
            deliverable[level] = float(room.sum() * STEP_H)
        met = [lvl for lvl, kwh in deliverable.items() if kwh >= need]
        confidence = max(met) if met else 0.0
        accepted = confidence >= 0.9
        answer = {"requirement": args.model_dump(), "energy_needed_kwh": round(need),
                  "deliverable_kwh_without_new_peak": {f"p{int(k * 100)}": round(v) for k, v in deliverable.items()},
                  "probability_met_without_new_peak": f">= {int(confidence * 100)}%" if confidence else "< 50%",
                  "accepted": accepted}
        if not accepted:
            short = need - deliverable[0.9]
            hours = steps * STEP_H
            extra_kw = short / hours if hours else float("inf")
            answer["to_guarantee_it"] = (f"raise the demand ceiling by about {extra_kw:.0f} kW "
                                         f"(about ${extra_kw * self._demand_rate(start):,.0f} this month) or lower the target to "
                                         f"{deliverable[0.9] / args.vehicles:.0f} kWh per vehicle")
        else:
            self.requirements.append({**args.model_dump(), "probability": answer["probability_met_without_new_peak"]})
        return answer

    def _demand_rate(self, t: int) -> float:
        season = self.spec.season_of(int(self.scenario.index[t].tz_convert(LOCAL_TZ).month))
        return sum(c.usd_per_kw for c in season.demand)

    def simulate_expansion(self, args: SimulateExpansion) -> dict:
        for split in ("test", "dev"):
            path = RESULTS / f"capacity_summary_{split}.csv"
            if path.exists():
                answer = expansion_answer(pd.read_csv(path), self.n_ports + args.extra_ports)
                answer["evaluated_on"] = "summer 2017 (held-out test months)" if split == "test" else "development months"
                answer["chargers_today"] = self.n_ports
                return answer
        raise RuntimeError("capacity study not run yet (scripts/capacity_study.py)")

    def grid_event_adjustment(self, args: GridEventAdjustment) -> dict:
        t0, t1 = self.step_of(args.start_local), self.step_of(args.end_local)
        issue = self.step_of(self.now_local)
        h0, h1 = max(0, t0 - issue), min(self.forecast.shape[1], t1 - issue)
        if h1 <= h0:
            return {"accepted": False, "reason": "the window is not within the next 24 hours of forecasts"}
        q = self.forecast[issue, h0:h1]
        lo = float((q[:, 0] / q[:, 1]).min())
        hi = float((q[:, 3] / q[:, 1]).max())
        accepted = lo <= args.building_multiplier <= hi
        result = {"proposal": args.model_dump(), "calibrated_band_multiplier": [round(lo, 3), round(hi, 3)],
                  "accepted": accepted}
        if accepted:
            self.adjustments.append(args.model_dump())
            result["effect"] = (f"planning load raised to x{args.building_multiplier:.2f} of the median forecast "
                                f"for {args.start_local}..{args.end_local}; the safety layer is unchanged")
        else:
            result["reason"] = ("outside the forecast's calibrated band: history doesn't support it, so it needs "
                                "a human decision before it can change the plan")
        return result

    def read_tariff(self, args: ReadTariff) -> dict:
        from .tariff_reader import bill_check, extract, field_accuracy

        pdf, gold_name = TARIFF_DOCS[args.document]
        spec, _ = extract(DOCS / pdf)
        out = {"document": pdf, "extracted": spec.model_dump()}
        gold_path = DOCS / "gold" / f"{gold_name}.json"
        if gold_path.exists():
            gold = load_gold(gold_name)
            acc = field_accuracy(spec, gold)
            bills = bill_check(spec, gold, self.site["building_kw"])
            out["verification"] = {"fields_correct": f"{acc['correct']}/{acc['fields']}", "wrong_fields": acc["wrong"],
                                   "bill_reproduction_worst_month_error": round(bills["worst_error"], 5),
                                   "passed": bills["passed"]}
        return out

    def grid_alerts(self, args: GridAlerts) -> dict:
        day = str(pd.Timestamp(args.local_date).date())
        if day in SITE.flex_alerts:
            hours = SITE.flex_alerts[day]
            return {"date": day, "alert": "CAISO Flex Alert (statewide)", "hours": hours or "not published",
                    "source": "CAISO Flex Alert archive (frozen event calendar, docs/DATA_SPEC.md)"}
        return {"date": day, "alert": None, "source": "CAISO Flex Alert archive (frozen event calendar)"}

    def comparison(self) -> pd.DataFrame:
        rows = [evaluate(r, self.scenario, self.spec) for r in self.results.values()]
        return pd.DataFrame(rows).set_index("controller")
