# Headroom

**More EV chargers on the grid connection a site already has: no service upgrade, and a tested answer to "what will the next 12 chargers cost me?"**

Meta Global AI Developer Hackathon 2026 · Indie track · *Push the Frontier* (energy: controlling a physical system)

At a mall, office or depot the chargers share one meter with the building, and US utilities bill the highest 15 minutes of each month. Under Palo Alto's E-4 TOU tariff, one 50 kW charging spike at 6 pm on a summer weekday adds **$2,572** to the bill, for about $3 of electricity. So owners stop adding chargers, or cap them and waste the capacity they already pay for. Upgrading the connection instead typically means a utility study, a new transformer and a long wait.

Headroom charges the cars in the room the building leaves:

1. **See the building.** A forecast of the building's load for the next 24 hours, as calibrated quantiles (an hour-of-day statistical profile; no trained model at runtime).
2. **Plan the charging.** A linear program (HiGHS) re-solved every 15 minutes charges every car under the demand level the month has already paid for, and raises it only by the minimum the drivers' needs require.
3. **Guarantee the limit.** A deterministic safety layer that nothing above it can override keeps the site under its connection limit (fuzz-tested on 10,000 random and adversarial plans).
4. **Muse Spark runs the site.** It reads the utility's tariff, takes requests by text or voice, sizes a plan change for a grid alert from the building's own past alert days, spots blocked bays on camera, and answers expansion questions from simulation. It can act only through validated tools, and every number in its answers is checked against the tool results. If the internet drops, **Muse Glimmer** on the site's own GPU takes over; charging never depends on either model.

> The model proposes, the math decides, the safety layer guarantees.

## Results
Held-out test months (May–Oct 2017), real data only; nothing was tuned on them. Same real driver sessions for every controller; 6 months × 3 resampled fleets = 18 month-scenarios per row.

**24 chargers (4× today's garage) on a 1,000 kW connection, E-4 TOU tariff** (`scripts/final_benchmark.py`):

| Controller | Energy delivered | Demand charges added by charging ($/month) | Connection-limit violations |
|---|---|---|---|
| Static cap | 100% | 4,231 | 0 |
| Least-laxity-first (standard smart charging) | 100% | 4,231 | 0 |
| Caltech adacharge MPC (open source, unmodified) | 100% | 4,231 | 0 |
| **Headroom** | **100%** | **2,889 (−32%)** | **0** |
| *Ablation: perfect foresight (upper bound)* | *100%* | *2,595* | *0* |

- **Hosting capacity** (drivers get ≥ 99% of their energy in every month, 0 violations; `scripts/capacity_study.py test`): **48 chargers with Headroom vs 36 with standard smart charging or a static cap (+33%)** on the same connection. At 48, standard smart charging leaves drivers short in 3 of the 18 month-scenarios.
- Headroom captures **82%** of the saving a perfect-foresight controller gets.
- **Stress test** (labelled; added intra-hour variability, σ = 2%): $2,783/month (standard smart charging: $3,903), 100% delivered, 0 violations.
- **Drivers first, as a strict priority.** The planner prices unserved energy above the most a kWh could ever save in demand charges, derived from the tariff (E-4: $309/kWh), so it never trades a driver's energy for a lower bill.
- **Ablations, reported as measured:**
  - A trained LightGBM forecaster in place of the profile changes almost nothing ($2,889.62 vs $2,888.91): the two forecasts agree for the first 4 hours, which is what drives each 15-minute decision. The value is in seeing the building, not in model complexity ([docs/FORECASTING.md](docs/FORECASTING.md)).
  - "Never set a new peak" adds only $326/month but delivers 86% of energy: some charging has to overlap the building's peak, which is why drivers come first.
- The three EV-only baselines coincide at 24 chargers because their fixed 161 kW cap never binds; they separate at higher counts.
- **Forecast calibration** on the test months: P90 covered 86–95% (target 90%), P99 97.7–100% (target 99%).

### Beyond one site: 49 real buildings, median +15% chargers
Headroom Check on 49 metered buildings from BDG2 at 7 US sites (22 education, 8 office, 7 lodging, 5 public services, 4 assembly, 2 parking, 1 services). They are a quality-filtered, fixed-seed sample, screened on held-out Jul–Aug 2017 with real Palo Alto sessions (`scripts/portfolio_study.py`):

| | Result |
|---|---|
| Buildings where Headroom fits more chargers on the same connection | **77%** (fewer in none) |
| Median gain per building | **+15%** (95% bootstrap CI +10% to +45%; IQR +2% to +61%) |
| Chargers in total, standard smart charging → Headroom | 2,225 → 3,010 (+35%) |
| Grid-limit violations at the reported charger counts | **0** |

- Gains vary widely by building: no change in 10 of 44, +50% or more in 14, and more than double in 6 (e.g. 22 → 76 chargers).
- In 5 small or very spiky buildings neither controller can guarantee room for even 2 chargers.
- Honest caveats:
  - In 6 buildings the test months contained a load rise larger than any in the building's past year (16 intervals in all). Fleet sizes where this caused a violation were rejected (6 simulated months).
  - The connection size is estimated, as stated under *Assumptions and limits*.
  - Our main site (+33%) is excluded from these figures.

## Meta models
| Model | Where it runs | Job in Headroom | How its output is checked |
|---|---|---|---|
| **Muse Spark 1.3** | Meta Model API | Site operator (tool-calling agent); tariff document → structured rates; grid-alert sizing; bay video → occupancy | Tool calls validated against strict schemas; every number in an answer traced to a tool result; tariff accepted only if it reproduces the real bills; bays count as blocked only if the charger also reports no power |
| **Muse Voice Transcribe 1.0** | Meta Model API | Spoken requests → text for the operator | The transcript is shown to the user before anything runs |
| **Muse Glimmer** (open weights, Apache-2.0) | The site's own GPU (Ollama); demo answers recorded on a Kaggle T4 | Backup operator when the internet link is down | Same tools, same validation, same number check |

Every model called at runtime is a Meta model; the planner (a linear program) and the forecast (a statistical profile) are not models. Each recorded model call (request, response, tokens, time) is in [`data/recordings/`](data/recordings/) and listed on the app's *Proof* page; the demo replays them, so it runs without any account or key.

## Run the demo (no key, no account)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
- **Overview** tells the story; **Live replay** shows the real heat-wave day; **Ask the operator** replays the recorded conversation one click per question; **Check any building** screens a sample building (or your own meter export); **How the AI is checked** and **Proof and data** show the verifiers and every result.
- Sidebar: **Cut the site's internet** switches the operator to Muse Glimmer; **Model calls: Live** calls the models (needs `MODEL_API_KEY` for the Meta Model API in `.env`; see `.env.example`). Keys live only in `.env`, which is never committed. A hard spend guard in `headroom/meta/client.py` refuses any call that could take the total past the budget.
- The first visit to a replay page simulates the month (about a minute) and caches it.

## Data (real recorded data only)
| Source | Use |
|---|---|
| City of Palo Alto open data: WEBSTER garage charging sessions, 2016–2017 (6 ports) | Every car's arrival, departure and energy. Larger garages re-use real recorded port-days (same day type, ±7 days) |
| Building Data Genome 2 (Miller et al., *Scientific Data* 7, 368, 2020): UC Berkeley `Bear_education_Patricia` | The building sharing the connection; hourly → 15-minute with an energy-preserving split |
| BDG2, other US sites | The multi-building study (quality-filtered, fixed-seed sample) |
| City of Palo Alto Utilities E-4 TOU and E-7 TOU tariffs (2026) | Bills; hand-checked gold copies for the tariff verifier |
| CAISO Flex Alert dates | Grid-stress events |

Preprocessing only transforms real data (time zones, cleaning, ≤ 3 h gap filling, the energy-preserving split). Details and every count: [docs/DATA_SPEC.md](docs/DATA_SPEC.md), `data/processed/REPORT.md`.

## Assumptions and limits
- Simulation on recorded data, not a live site: the building and the garage are real but from neighbouring cities, sharing an assumed 1,000 kW connection.
- 2026 prices on 2016–2017 behaviour; one tariff family (CPAU). The tariff reader is built to take other utilities' documents.
- In the multi-building study the connection size isn't in the data: estimated as the peak before the screened months ÷ 0.8 (the electrical code's continuous-load rule).
- The simulation steps in 15 minutes, so the safety layer keeps an allowance for the building's load rising before the next reading. It budgets against the building's highest reading of the past 3 hours (riding through meter dropouts) plus an allowance sized from that building's own past year (largest rise + 20%; 60 kW for the main building). The guarantee holds whenever a rise stays within that allowance, and every run reports any rise that didn't. A real deployment would read the meter every few seconds.
- The bay camera demo uses public-domain parking-lot footage with bays marked for illustration.

## Built during the hackathon, and what isn't ours
**Built during the hackathon (Oct 26 – Nov 4, 2026; git history from Oct 27):** everything in `headroom/`, `scripts/`, `tests/` and `app.py`: data pipeline, simulator, safety layer, Headroom's planner, forecasts, transformer model, tariff reader and verifiers, operator agent and tools, number check, failover, Headroom Check and the studies, and the app.

**Third-party, used unmodified as libraries:** [adacharge](https://github.com/caltech-netlab/adacharge) and [acnportal](https://github.com/zach401/acnportal) (Caltech; the adacharge MPC baseline), HiGHS (via SciPy), CVXPY/Clarabel (for adacharge), LightGBM (ablation only), Streamlit, Plotly, pandas, NumPy, pydantic, the OpenAI Python SDK (as the client for Meta's OpenAI-compatible API).

**Prior work:** the forecasting idea (an hour-of-day baseline improved by gradient boosting) follows our 2025 IndiaAI hackathon entry; no code from it is used. Our first commit (`b85b2e1`) accidentally included that 2025 notebook and its write-up; they were removed in `cc2d789` and nothing in Headroom uses them.

## Reproduce the results (Windows, Python 3.12)
```powershell
.\.venv\Scripts\python.exe scripts\fetch_data.py                 # raw data (BDG2 is 167 MB)
.\.venv\Scripts\python.exe -m headroom.data.pipeline             # data/processed (~5 s)
.\.venv\Scripts\python.exe -m pytest -q                          # invariants, safety fuzz test, agent and verifier tests
.\.venv\Scripts\python.exe scripts\precompute_forecasts.py 2017-05 2017-06 2017-07 2017-08 2017-09 2017-10
.\.venv\Scripts\python.exe scripts\final_benchmark.py            # the table above (~20 min)
.\.venv\Scripts\python.exe scripts\capacity_study.py test        # hosting capacity (~25 min)
.\.venv\Scripts\python.exe scripts\portfolio_study.py select     # the multi-building sample
.\.venv\Scripts\python.exe scripts\portfolio_study.py run        # the multi-building study (hours)
```

## Code layout
- `headroom/data/`: datasets → one 15-minute site table
- `headroom/tariff.py`: tariffs as data, and the monthly bill
- `headroom/sim/`: scenarios on real sessions; the 15-minute simulator with the safety layer
- `headroom/control/`: Headroom's planner and the baselines
- `headroom/forecast/`: hour-of-day profile (runtime), LightGBM blend (ablation), walk-forward files
- `headroom/check.py`: Headroom Check for any building's meter data
- `headroom/thermal.py`: IEEE C57.91 transformer ageing
- `headroom/agent/`: operator agent, tools, number check, tariff reader, bay check, the replay site
- `headroom/meta/client.py`: Meta model client with record/replay, failover and the spend guard
- `scripts/`: benchmark, capacity and multi-building studies, forecasts, demo recording, data download
- `docs/`: [FORECASTING](docs/FORECASTING.md), [DATA_SPEC](docs/DATA_SPEC.md), [VIDEO](docs/VIDEO.md), [GLIMMER_KAGGLE](docs/GLIMMER_KAGGLE.md)
