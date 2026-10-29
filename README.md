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

## Results (held-out test months, May–Oct 2017; nothing was tuned on them)
Same real sessions for every controller; 6 summer months × 3 resampled fleets = 18 month-scenarios per row. Reproduce with `scripts/final_benchmark.py` and `scripts/capacity_study.py test`.

**24 chargers (4× today's garage), E-4 TOU tariff:**

| Controller | Energy served | Demand charges added by EVs ($/month) | Service-limit violations |
|---|---|---|---|
| Static cap | 100% | 4,231 | 0 |
| Least-laxity-first (ACN-Sim) | 100% | 4,231 | 0 |
| Caltech adacharge MPC (unmodified) | 100% | 4,231 | 0 |
| **Headroom** | **99.8%** | **2,670 (−37%)** | **0** |
| *Perfect-foresight ceiling (ablation)* | *99.5%* | *1,979* | *0* |

- **Hosting capacity** (every month ≥ 99% of energy delivered, 0 violations): **48 chargers with Headroom vs 36** for the baselines on the same 1,000 kW connection (**+33%**).
- Headroom captures **69%** of the savings a perfect-foresight controller could get.
- Holds under a labelled stress test with added intra-hour variability: $2,637/month, 99.8% served, 0 violations.
- Honest ablations: profile-only forecasting does as well in the control loop ($2,676): the savings come from *seeing the building*, not from model complexity. The strict "never set a new peak" policy adds only $326 but serves 86% of energy: some charging must overlap the building's peak.
- The three EV-only baselines coincide at 24 chargers because their fixed 161 kW cap never binds; they separate at 48+.
- Forecast calibration on the test months: P90 covered 87–92% (target 90%), P99 98–100% (target 99%).

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

## Run the demo (no API key needed)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
The control room opens with **Model calls: Replay recorded**: the operator, tariff and bay answers are real Muse Spark and Muse Glimmer responses we recorded (`data/recordings/`), so the full demo runs without credits. Ask the requests listed in `scripts/record_demo.py` to replay them; any other request needs Live mode and a key. The first load simulates the month (about a minute) and caches it. Flip **Cut the site's internet link** to see the operator fail over to Muse Glimmer.

**Live mode:** `copy .env.example .env`, add a key (Meta Model API, or a partner serving Muse Spark Open / Glimmer), and switch Model calls to Live. Keys live only in `.env`, which is never committed.

## Reproduce the results (Windows, Python 3.12)
```powershell
.\.venv\Scripts\python.exe scripts\fetch_data.py               # restores the 167 MB BDG2 file
.\.venv\Scripts\python.exe -m headroom.data.pipeline           # builds data/processed (~5 s)
.\.venv\Scripts\python.exe -m pytest -q                        # invariants, safety fuzz test, agent validation
.\.venv\Scripts\python.exe scripts\precompute_forecasts.py 2017-05 2017-07 2017-08 2017-09 2017-10   # walk-forward forecasts (2017-06 ships with the repo)
.\.venv\Scripts\python.exe scripts\final_benchmark.py                 # held-out benchmark table above
.\.venv\Scripts\python.exe scripts\capacity_study.py test             # hosting capacity
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
