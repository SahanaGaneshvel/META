# Headroom

**Add EV chargers to the grid connection a site already has, without a service upgrade, and know in advance what each expansion will cost.**

Meta Global AI Developer Hackathon 2026 · Indie track · *Push the Frontier* (energy: controlling a physical system).

At a mall, office or depot, the EV chargers share one electricity meter with the building. US utilities bill the highest 15-minute demand of the month: under Palo Alto's E-4 TOU tariff, one 50 kW spike at 6 pm on a summer weekday adds **$2,572** to the bill, about 950× the cost of the energy it delivered. So owners stop adding chargers, or cap them statically and waste capacity.

Headroom runs the chargers inside the room the building leaves:
- a **probabilistic forecaster** of the building's load (calibrated quantiles, re-calibrated daily);
- a **planner** (an LP on HiGHS) that charges every car under the demand level the month has already paid for, and raises it only by the minimum the physics requires;
- a **deterministic safety layer** that nothing above it can override (fuzz-tested on 10,000 random and adversarial plans);
- **Muse Spark Open as the site's operator**: it reads the utility's tariff, takes requests by text or voice, watches the bays, and answers "can I add 12 more chargers?" from simulation on real sessions. Every number it gives comes from a tool; every tool call is validated. **Muse Glimmer** takes over if the network drops.

> "The model proposes, the math decides, the safety layer guarantees."

## Results
Final numbers come from `scripts/capacity_study.py test` and the benchmark scripts on the held-out test months (May–Dec 2017) and are written to `data/results/`. Development numbers (train/validation months only) are in `data/results/dev_*.csv`.

## Meta models used (all open weights)
| Model | Where it runs | What it does in Headroom |
|---|---|---|
| **Muse Spark Open** | Partner inference (Baseten / Fireworks / Modal; OpenAI-compatible) | Operator agent with validated tool calls; tariff document → strict JSON, verified by reproducing real bills; voice requests (audio input); bay video → blocked bays |
| **Muse Glimmer** | Partner inference | Offline backup operator |
| **Muse Code** | Development | Coding agent during the build |

Every model the project calls at runtime is a Meta model. Models we train ourselves (the LightGBM quantile forecaster) are our own code. Model responses used in the demo are recorded (`data/recordings/`) so the hosted demo can replay them.

## Data (real recorded data only)
Palo Alto WEBSTER garage charging sessions (2016–2017), a UC Berkeley building's metered load (Building Data Genome 2), station weather, Palo Alto's published E-4 TOU and E-7 TOU tariffs, and CAISO's real Flex Alert dates. Preprocessing only transforms real data (time zones, cleaning, ≤3 h gap filling, an energy-preserving hourly → 15-minute split). Larger garages re-use real recorded sessions on the extra ports. Everything is in **[docs/DATA_SPEC.md](docs/DATA_SPEC.md)**, with every count in `data/processed/REPORT.md`.

## Built during the hackathon (Oct 26 – Nov 4, 2026)
Everything in this repository: the data pipeline, simulator, baselines harness, forecaster, planner, safety layer, transformer model, tariff reader and verifiers, agent and tools, capacity study and control room. The git history starts on Oct 27, 2026.

## Prior work
The forecasting idea (an hour-of-day baseline improved by gradient boosting) follows our 2025 IndiaAI hackathon entry; all code here was written during this hackathon.

## Run it (Windows, Python 3.12)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
copy .env.example .env                                         # add a partner key for Muse Spark Open
.\.venv\Scripts\python.exe scripts\fetch_data.py               # restores the 167 MB BDG2 file
.\.venv\Scripts\python.exe -m headroom.data.pipeline           # builds data/processed (~5 s)
.\.venv\Scripts\python.exe -m pytest -q                        # invariants, safety fuzz test, agent validation
.\.venv\Scripts\python.exe scripts\precompute_forecasts.py 2017-06   # walk-forward forecasts for a month
```

## Code layout
- `headroom/data/`: datasets → one 15-minute site table
- `headroom/tariff.py`: tariffs as data and the monthly bill
- `headroom/sim/`: scenarios on real sessions and the 15-minute simulator with the safety layer
- `headroom/control/`: Headroom's planner and the baselines (static cap, least-laxity-first, Caltech's adacharge MPC)
- `headroom/forecast/`: hour-of-day profile, LightGBM quantiles, rolling conformal calibration, walk-forward files
- `headroom/thermal.py`: IEEE C57.91 transformer ageing
- `headroom/agent/`: operator agent, tools, tariff reader and verifiers, the replay site
- `headroom/meta/client.py`: one client for Meta's open-weight models on any partner, with record/replay
- `scripts/`: sweeps, capacity study, forecast precompute, data download
- `docs/`: strategy, plan, data specification
