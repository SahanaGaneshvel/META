# Project context — Headroom (Meta Global AI Developer Hackathon 2026)

Context for anyone (or any coding assistant) working on this repo, on any machine. Read docs/STRATEGY.md, docs/PLAN.md and docs/DATA_SPEC.md before starting work.

## The project
- **Headroom**: an AI operator that fits more EV chargers on a site's existing grid connection. **Indie Developer track**, theme "Push the Frontier" (energy). Owner: Sahana Ganeshvel.
- **Goal: win first place.** Judging weights: Impact 30, Use of Meta AI tooling 25, Demo 25, Technical depth 20. Weigh every decision by its effect on these.
- Dates (IST): kickoff Oct 26, 6:30 PM. **Submission deadline Nov 5, 7:30 AM IST** (Nov 4, 9 PM EST); aim to submit by Nov 4, 10 PM IST. **Core freeze Nov 1; feature freeze Nov 3.** Day-by-day plan: docs/PLAN.md.
- Follow the Official Rules. The IndiaAI notebook (2025) is prior work and is disclosed in the README.

## Decisions already made (don't relitigate without new evidence)
- **US-first.** Site = Palo Alto WEBSTER garage (6 ports) 2016–2017 + UC Berkeley building `Bear_education_Patricia` (BDG2) + Palo Alto **CPAU E-4 TOU** tariff (2026). India (Tamil Nadu) is **future scope only**.
- Headline metric: **probabilistic hosting capacity** (extra chargers at <1% risk of a new monthly 15-min demand peak). Beat **strong baselines**: static cap, LLF and adacharge MPC (both already get 0 violations, so "zero violations" alone isn't novel).
- The US hook: a +50 kW spike for 15 min at 6 pm in summer adds **$2,572** to the monthly bill (two demand charges, $30.18 + $21.27/kW), about **950×** the $2.69 the energy costs.
- **Demo replay day = 2017-06-20 and 2017-06-21 (real CAISO Flex Alert heat-wave days).** Building 4–9 pm peak +10% (799 vs 728 kW). Then a live cut (Open-Meteo weather, CAISO grid demand).
- **Stanford game days were tested and showed NO effect (0.97×). Never claim a game-day surge.**
- Every Meta-model job has a verifier: tariff document → JSON → bill reproduction; grid alerts → bounded adjustment; bay video → blocked-bay detection checked against charger data (28.2% of plugged-in time is idle: measured).
- "The LLM proposes, the math decides, the safety layer guarantees."

## Data (authoritative spec: docs/DATA_SPEC.md)
- Rebuild the processed tables with `python -m headroom.data.pipeline` (~5 s); tests: `python -m pytest -q`; report in `data/processed/REPORT.md`.
- Splits: train 2016; val 2017-01..04; **test 2017-05..12, untouched until the day-9 benchmark.**
- `web_search` is **off** in backtests (Meta search has no date filter, so it would leak the future).
- Weather: train on the BDG2 station; calibrate live Open-Meteo to it (r = 0.912, so not interchangeable).

## Meta models (open weights only; revised 2026-10-29)
- **No Meta Model API**: it needs a payment card, which the owner can't add (Meta confirmed this doesn't count against us). So no hosted muse-spark-1.3, SAM 3.1, Voice Transcribe or built-in `web_search`.
- **Muse Spark Open** (816B MoE, 42B active; text, image, audio and video in; 1M context) and **Muse Glimmer** (30B, Apache-2.0; text and image in) through partner inference. All OpenAI-compatible:
  - Baseten: `https://inference.baseten.co/v1`, hackathon model id **`meta-models/muse-spark-open-1026`**; supports tools, `tool_choice` (incl. required), `response_format` json_schema, `reasoning_effort`. No PDF input (send text or page images); ≤8 images and 2 videos per request.
  - Fireworks: Glimmer is `fireworks/muse-glimmer-30b`. Modal Endpoints: proxy token `wk-…/ws-…`.
- Provider, base URL, key and model ids live in `.env` (see `.env.example`). Every model response used in the demo is recorded and replayable.
- Grid-event search is our own tool set (CAISO alerts, Open-Meteo), off in backtests (frozen event calendar instead).
- **Never paste API keys into chat; they go in `.env` only.**

## Environment gotchas
- Python **3.12**. `acnportal` needs `setuptools<80` (pinned). `adacharge` installs from GitHub (needs git).
- Windows console is cp1252: run with `PYTHONIOENCODING=utf-8` or keep output ASCII.
- pandas 3: use `day_of_week` (not `dayofweek`); mixing datetime units raises errors.
- No GPU needed locally: models run through partner inference; LightGBM and the LP planner run on CPU.
- `data/raw/bdg2/electricity_cleaned.csv` (167 MB) is gitignored because of GitHub's 100 MB file limit: `python scripts/fetch_data.py` restores it.

## Working style the owner expects
- Be honest: say what can't be promised, test claims against data before using them, and report negative results.
- Explain choices in terms of the judging criteria. Keep docs (PLAN, DATA_SPEC, STRATEGY) updated the same day decisions change.
