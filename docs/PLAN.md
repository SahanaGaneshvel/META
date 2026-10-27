# Headroom — 10-day build plan

All times are IST (you're in Chennai). US daylight saving ends Nov 1, so:
- Kickoff: **Oct 26, 9:00 AM ET = 6:30 PM IST**
- Submission deadline: **Nov 4, 9:00 PM EST = Nov 5, 7:30 AM IST.** Plan to submit by **Nov 4, 10 PM IST**; don't rely on the overnight margin.

## Setup checklist

- [ ] *(Optional; not blocking since the switch to US-first)* ACN-Data API token from https://ev.caltech.edu/register (the registration form was returning HTTP 500; retry once). Without it, the fleet scenario uses Palo Alto sessions with actual departure and delivered kWh as the driver's request (standard practice when driver inputs aren't recorded)
- [x] Palo Alto tariff PDFs downloaded to `data/docs/` (CPAU E-4 TOU and E-7 TOU effective 2026-07-01; E-4 effective 2025-07-01)
- [ ] Hand-check the gold labels for E-4 TOU (`data/docs/gold/`): every rate, window and rule against the PDF
- [x] Building Data Genome 2 downloaded (UC Berkeley buildings, 2016–2017); site = Palo Alto WEBSTER sessions 2016–2017 (see DATA.md)
- [x] Data audit done and dataset choices fixed: **docs/DATA_SPEC.md** is the authoritative data spec (sources, evidence, preprocessing, splits, assumptions)
- [x] Event calendar: Stanford games tested (no effect, so dropped); CAISO Flex Alert days chosen (measured effect)
- [x] Parking-lot video: two static-camera clips (public domain, Wikimedia Commons) in `data/video/`, 5 fps H.264; see `data/video/SOURCES.md`
- [ ] Locate your IndiaAI hackathon forecasting code and notebooks (prior work, to reuse if the rules allow it)
- [ ] Oct 26: join the hackathon Discord (invite only arrives at kickoff); turn on notifications for #announcements
- [ ] Apply for the Baseten, Modal and Fireworks credits as soon as the forms open
- [ ] **Oct 26: read the Official Rules first**, especially the parts on pre-existing code, team size, submission format and video length
- [ ] Oct 26: put your API key in `.env` as `MODEL_API_KEY`, then run `python scripts/api_smoke.py`
- [ ] Build in public: post on LinkedIn/X on day 1, day 5 and day 10 (there are bonus prizes for community vote and most viral project)

## Day-by-day

| Day | Date (IST) | Goal | Done when |
|---|---|---|---|
| 1 | Mon Oct 26 (evening) | Rules, API key, smoke tests pass; SAM video format confirmed | All 5 smoke tests green |
| 2 | Tue Oct 27 | **Data pipeline** (`headroom/data`, done) + **site simulator** (1,000 kW service, 15-min demand, transformer thermal, N-port scale-up, intra-hour noise) and four baselines | Baseline table for one site-month |
| 3 | Wed Oct 28 | **Forecaster**: port the IndiaAI ensemble, add quantiles (LightGBM quantile + conformal calibration), walk-forward backtest | P10/P50/P90 coverage within ±3% of nominal |
| 4 | Thu Oct 29 | **Controller**: MPC (cvxpy + HiGHS) on P90 headroom + safety layer; property/fuzz tests | 10,000 fuzzed plans, 0 violations |
| 5 | Fri Oct 30 | **Agent**: tool schemas, loop, validators; CPAU tariff PDF → `TariffSpec` → bill check (E-4 TOU, then E-7 TOU to show it generalises) | Bill reproduced within ±1% |
| 6 | Sat Oct 31 | **Hosting capacity**: Monte Carlo "add N chargers" + agent tool; plain-language requests + explanations | Agent answers the expansion question end to end |
| 7 | Sun Nov 1 | **SAM 3.1 blocked bays** + **live mode** (Open-Meteo weather, CAISO grid demand and Flex Alerts, search). **Feature freeze tonight** | Blocked bay flagged on a test clip; the live panel shows today's real data |
| 8 | Mon Nov 2 | **Control-room UI** (Streamlit) + OCPP output; full demo dry run ×3 | The demo script runs without touching code |
| 9 | Tue Nov 3 | Final benchmarks, README, architecture figure, **record the video** | Video exported; README complete |
| 10 | Wed Nov 4 | Buffer: fix, re-record if needed, **submit by 10 PM IST** | Submission confirmation email |

## Daily rules
- Work on `main` with small commits; tag `day-N` each night.
- Every result that goes in the video comes from a script in `scripts/` that regenerates it.
- Stop each night with the demo still in working order.
