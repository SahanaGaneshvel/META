# Headroom

Fit more EV chargers on the grid connection a site already has.

Meta Global AI Developer Hackathon 2026, Indie track, theme: *Push the Frontier* (energy).

> Status: **pre-hackathon setup** (environment, data, design, API smoke tests). Product code starts at kickoff on Oct 26, 2026.

- Strategy and judging fit: [docs/STRATEGY.md](docs/STRATEGY.md)
- 10-day plan and checklist: [docs/PLAN.md](docs/PLAN.md)
- **Data specification (datasets, evidence, preprocessing, splits): [docs/DATA_SPEC.md](docs/DATA_SPEC.md)**
- Data inventory and audit log: [docs/DATA.md](docs/DATA.md)

Market: **United States first** (Palo Alto site, CPAU E-4 TOU tariff). India (Tamil Nadu) is future scope; the agent reads any utility's tariff.

## Setup (Windows, Python 3.12)

Full step-by-step guide for a new laptop: **[docs/SETUP.md](docs/SETUP.md)**. Project context, decisions and gotchas: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
copy .env.example .env   # then fill in MODEL_API_KEY
.\.venv\Scripts\python.exe scripts\fetch_data.py         # restores the 167 MB BDG2 file (over GitHub limit)
.\.venv\Scripts\python.exe -m headroom.data.pipeline       # builds data/processed (~5 s)
.\.venv\Scripts\python.exe -m pytest -q                     # data and parser tests
.\.venv\Scripts\python.exe scripts\api_smoke.py              # needs MODEL_API_KEY
```

## Code layout
- `headroom/data/`: loads the public datasets and aligns them into one 15-minute site table (`python -m headroom.data.pipeline`; report in `data/processed/REPORT.md`)
- `headroom/meta/`: Meta Model API client and SAM 3.1 output parser
- `scripts/`: API smoke tests, baseline check, data download, video clip cutting
- `tests/`: invariants (physical limits, DST handling, energy preservation, splits)

## Prior work (disclosed)
- Forecasting approach (Hour-of-Day baseline + TinyTimeMixer + per-horizon LightGBM, stacked and calibrated) from the IndiaAI Impact Gen-AI Hackathon 2025, Track 2 (1st place), by Sahana Ganeshvel and Vijay Bala Mahalingam.
