# Data sources and findings

> **The authoritative choices, preprocessing and splits are in [DATA_SPEC.md](DATA_SPEC.md).** This file is the inventory and audit log.

| Dataset | Use | Status | Notes |
|---|---|---|---|
| City of Palo Alto EV sessions 2011–2020 | **Mall scenario**: short public sessions | ✅ `data/raw/paloalto_ev_sessions.csv` (259,415 rows, 85 MB) | Source: https://data.paloalto.gov/datasets/194693 . No license stated by the publisher, so cite it as public open data. **40 End Date values are Excel serial numbers** (e.g. `42845.81`) and 43 rows have end ≤ start; parse with the helper below |
| Caltech ACN-Data (Caltech, JPL, Office001) | **Workplace/fleet scenario**: long stays, user-entered requested kWh and departure | *Optional.* Not used: the registration form returned HTTP 500 and no public mirror exists (the DOI 10.57702/vtc26i8u record has no files) | Data from the Python client is free of the web-UI timezone bug |
| **CPAU E-4 TOU** (effective 2026-07-01) | **Primary tariff** the agent reads; bill engine; gold labels | ✅ `data/docs/e-4-tou_effective_2026-07-01.pdf` (5 pages) | Medium commercial, 500–1,000 kW. Summer: peak demand $30.18/kW (4–9 pm weekdays), max demand $21.27/kW, energy $0.21556 / $0.18393 / $0.14715 per kWh (peak/mid/off). Winter: both demands $13.80/kW, energy $0.16170 / $0.13506 / $0.10416. Customer charge $131.92/month. Demand = max **15-min** average kW per period. Summer runs May 1–Oct 31 |
| CPAU E-7 TOU (effective 2026-07-01) | Second tariff, to test whether extraction generalises | ✅ `data/docs/e-7-tou_effective_2026-07-01.pdf` | Large commercial (≥1,000 kW). Summer peak demand $28.96/kW, max demand $18.58/kW (per search summary; verify in the PDF) |
| CPAU E-4 (non-TOU, effective 2025-07-01) | Contains the **5-minute-interval clause** for fluctuating loads | ✅ `data/docs/e-4-effective-2025-07-01.pdf` | A detail the agent should flag |
| Building Data Genome 2 (CC-BY 4.0) | Uncertain building load sharing the meter | ✅ `data/raw/bdg2/` (metadata, hourly electricity 2016–2017 for 1,578 meters, weather) | Site "Bear" = UC Berkeley (US/Pacific, Bay Area climate). Candidates in the E-4 TOU range (500–1,000 kW): **Bear_education_Patricia** (mean 632, max 850 kW), Bear_education_Benita (522 / 732), and Bear_assembly_Genia (261 / 614, event-driven and peaky, so the forecasting is harder). Hourly only: 15-min load is derived (see the caveat below) |
| Event calendar 2016–2017 | Frozen, labelled demand shocks for backtests | ⏳ Stanford home football games, Festival of the Arts, commencement, US federal holidays | Measure the real effect before claiming it |

## Chosen site (aligned years and region)

**Palo Alto WEBSTER garage sessions, 2016–2017** (10,132 and 8,761 sessions, its busiest years) **+ a Berkeley building, 2016–2017 + CPAU E-4 TOU.** Same years, same region, one consistent site.

| Palo Alto sessions by year | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|
| WEBSTER | 10,132 | 8,761 | 6,879 | 6,545 |
| HIGH | 5,070 | 8,908 | 6,733 | 7,644 |
| BRYANT | 5,372 | 6,667 | 7,625 | 7,665 |
| CAMBRIDGE | 5,789 | 5,846 | 6,895 | 9,411 |

Caveat to state in the README: BDG2 is hourly. The 15-minute building load is the hourly value plus intra-hour variation, sampled from residuals fitted to the hourly ramps. EV load is exact at 15 minutes, because it comes from session times.
| *(Future scope)* TNERC EV tariff, Hyderabad station data | India extension | Not needed for the hackathon | News reports (Jul 2025): ₹6.50 solar / ₹9.75 peak / ₹8.10 night per kWh |
| Parking-lot video | SAM 3.1 blocked-bay demo | ✅ `data/video/lot_wide_5fps.mp4` (154 s) and `lot_close_5fps.mp4` (63 s) + stills | Public domain (Supercircuits, Wikimedia Commons). Static segments cut by motion analysis; 5 fps to save SAM per-frame cost; ≤16 tracked objects per frame, so use an 8–12-bay polygon. Pexels and Pixabay block scripted downloads (HTTP 403). See `data/video/SOURCES.md` |

## Palo Alto site profile (2019)

| Site | Sessions | Ports | Max concurrent | Median kWh | Median stay (h) |
|---|---|---|---|---|---|
| BRYANT | 7,665 | ~12 | 10 | 7.8 | 2.1 |
| HIGH | 7,644 | ~8 | 8 | 7.5 | 1.9 |
| WEBSTER | 6,545 | ~6 | 6 | 8.1 | 2.2 |
| CAMBRIDGE | 9,411 | ~10 | 10 | 5.8 | 1.6 |

Short, ~8 kWh stays look like a mall or downtown garage. ACN Caltech (long workplace stays) covers the fleet-depot scenario.

## Date parsing helper (pandas 3)

```python
def parse(col):
    num = pd.to_numeric(col, errors="coerce")
    txt = pd.to_datetime(col.where(num.isna()), format="mixed").dt.as_unit("s")
    xl = pd.to_datetime(num, unit="D", origin="1899-12-30").dt.round("s").dt.as_unit("s")
    return txt.where(num.isna(), xl)
```

## Environment gotchas
- `acnportal 0.3.3` imports `pkg_resources`, so it needs `setuptools<80` (pinned).
- ACN's MPC baseline isn't on PyPI: `pip install git+https://github.com/caltech-netlab/adacharge.git` (class `AdaptiveSchedulingAlgorithm`, pinned in requirements.lock.txt). Re-pin `setuptools<80` afterwards.
- ACN-Sim simulations mutate EV objects; `copy.deepcopy` the events and network for each run, or the second algorithm starts with full batteries.
- The Windows console uses cp1252: run with `PYTHONIOENCODING=utf-8`, or keep printed output ASCII-only.
- Meta `web_search` has **no date filter**, so a backtest must use a frozen event calendar (see STRATEGY §1 row 6).
