"""Confirm the open-source smart-charging baselines run in this environment.

    python scripts/baselines_check.py

Runs ACN-Sim's Caltech network with a small synthetic morning of EVs under three schedulers. Both
smart schedulers should report zero constraint violations: the bar Headroom has to beat elsewhere.
"""
from __future__ import annotations

import copy
import warnings
from datetime import datetime

import adacharge
import cvxpy
import numpy as np
import pytz
from acnportal import acnsim, algorithms

VOLTAGE = 208
PERIOD_MIN = 5


def synthetic_events(network, n_evs: int = 40, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    events = []
    for i, station in enumerate(rng.choice(network.station_ids, size=n_evs, replace=False)):
        arrive = int(rng.integers(90, 130))  # 07:30-10:50 in 5-minute periods
        leave = arrive + int(rng.integers(48, 110))
        ev = acnsim.EV(arrive, leave, float(rng.uniform(5, 25)), station, f"ev{i}", acnsim.Battery(100, 0, 100))
        events.append(acnsim.PluginEvent(arrive, ev))
    return events


def run(name: str, scheduler, network, events, start) -> None:
    # Simulations mutate EVs and the network, so every run gets fresh copies.
    sim = acnsim.Simulator(copy.deepcopy(network), scheduler, acnsim.EventQueue(copy.deepcopy(events)),
                           start, period=PERIOD_MIN, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run()
    violations = sum("Invalid schedule" in str(w.message) for w in caught)
    peak_kw = (acnsim.aggregate_current(sim) * VOLTAGE / 1000).max()
    delivered = acnsim.proportion_of_energy_delivered(sim)
    print(f"{name:14s} peak {peak_kw:6.1f} kW   energy delivered {delivered:.3f}   violation steps {violations}")


def main():
    start = pytz.timezone("America/Los_Angeles").localize(datetime(2019, 9, 5))
    network = acnsim.sites.caltech_acn(basic_evse=True, voltage=VOLTAGE)
    events = synthetic_events(network)
    solver = "ECOS" if "ECOS" in cvxpy.installed_solvers() else "CLARABEL"
    schedulers = {
        "uncontrolled": algorithms.UncontrolledCharging(),
        "least_laxity": algorithms.SortedSchedulingAlgo(algorithms.least_laxity_first),
        "adacharge_mpc": adacharge.AdaptiveSchedulingAlgorithm(
            [adacharge.ObjectiveComponent(adacharge.quick_charge),
             adacharge.ObjectiveComponent(adacharge.equal_share, 1e-12)], solver=solver),
    }
    for name, scheduler in schedulers.items():
        run(name, scheduler, network, events, start)


if __name__ == "__main__":
    main()
