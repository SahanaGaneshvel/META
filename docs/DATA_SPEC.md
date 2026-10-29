# Headroom — data specification (authoritative)

Decided after auditing the data. Rule: **every number in the video traces to a real dataset, and every assumption is listed here.** If code and this file disagree, fix one of them the same day.

## 0. Real data only (owner's rule, 2026-10-29)

- **Every input is real recorded data**: Palo Alto charging sessions, a UC Berkeley building's metered load, station weather, Palo Alto's published tariff PDFs, CAISO's real Flex Alert dates. No synthetic dataset is used anywhere in a headline number.
- **Preprocessing only transforms real data** (sections 3A–3D): parsing, time-zone handling, cleaning, gap filling of ≤ 3 h, and an energy-preserving hourly → 15-minute split. Each step and every count it changed is in `data/processed/REPORT.md`.
- **One unavoidable counterfactual**: "what if the garage had N ports?" The real garage had 6. Extra ports are filled by **re-using real recorded sessions** from neighbouring days (section 3E): every simulated car is a real car with its real arrival, departure and energy; only its placement on an extra port is resampled. N = 6 reproduces the real garage exactly.
- **Stress tests are labelled as such**: the optional intra-hour noise (3C step 4) is never on in a headline number.
- **Tariff fine-tuning data** (stretch goal) comes from real US tariffs (OpenEI Utility Rate Database), not generated documents.


## 1. Datasets chosen, with evidence

| Role | Dataset and exact subset | Why this one (measured) | Licence |
|---|---|---|---|
| **EV sessions** | City of Palo Alto, **WEBSTER** garage (3 stations × 2 ports = **6 Level-2 ports**), 2016-01-01 to 2017-12-31, **18,893 sessions** | The only busy site with a **constant port count for all 24 months** (HIGH changed between 4 and 8 ports, BRYANT between 5 and 12; CAMBRIDGE had a 13-day outage). 0 days without sessions, 0 duplicates, 100% Level 2. Measured charging power p99 = 6.18 kW, so the cap is **6.2 kW per port** | Public open data, no licence stated (cite the source) |
| **Building load** | Building Data Genome 2, **Bear_education_Patricia** (UC Berkeley), hourly kWh, 2016–2017 | Fewest gaps among the candidates (**41 of 17,544 hours missing, 0.2%**), no zeros or flat runs. Mean 632 kW, max 850 kW, so it fits **CPAU E-4 TOU (500–1,000 kW) without scaling**. Temperature-sensitive (r = +0.61), so the forecaster and live weather matter. Summer weekday 4–9 pm peak ≈ 98% of the daily peak, so the **peak-period demand charge is binding** | CC-BY 4.0 |
| Rejected building candidates | Rebecca (max 1,061 kW, beyond E-4 TOU), Genia (event-driven and small), Benita and Fannie (about 10× more gaps), Mara (smaller, less weather-sensitive) | — | — |
| **Weather (backtest)** | BDG2 weather, site Bear: hourly air temperature 2016–2017 | 100% coverage; local-time check passes (mean temperature peaks at 14:00) | CC-BY 4.0 |
| **Weather (live)** | Open-Meteo forecast API, Palo Alto (37.44, −122.14) | Verified working (no key). **Compared:** Open-Meteo history vs the BDG2 station over 17,544 h gives r = 0.912, bias −1.0 °C, RMSE 2.44 °C, below the 0.95 bar. **Decision: train on the BDG2 station; calibrate live Open-Meteo forecasts to it (linear map fitted on 2016–2017).** Both columns are kept in the site table | CC-BY 4.0 |
| **Tariff** | CPAU **E-4 TOU** effective 2026-07-01 (primary); E-7 TOU 2026 (generalisation test); E-4 2025 (5-minute clause) | Real, current, and the same utility as the site | Public tariff |
| **Grid-stress events** | CAISO **Flex Alert** days: 2016-06-20, 2016-07-27, 2016-07-28, **2017-06-20 and 2017-06-21 (2–9 pm)**, 2017-09-01 (1–10 pm) | Measured effect (section 2). The original notice PDFs are offline (404, not in the Internet Archive), so the backtest uses a **cited structured calendar**, never a reconstructed document | Public |
| **Holidays** | US federal holidays (`holidays` package) | Tariff "except holidays"; **verify against the CPAU Rules & Regulations definition on day 1** | — |
| **Camera** | Public-domain security-camera video, two static segments, 5 fps | See `data/video/SOURCES.md` | Public domain |

### Rejected after testing (reported in the README as negative results; it shows rigour)
- **Stanford home football games** (12 games, 2016–2017): **no effect** on Webster charging. 0.97× a same-weekday baseline all day, 0.96× after 3 pm. The "match night surge" demo moment is dropped.
- ACN-Data (registration broken), TNERC and Hyderabad (India is future scope).

## 2. The real demand shock: Flex Alert days

Same-weekday baseline = ±1–3 weeks around each date, excluding event days.

| Date | Max temperature (°C) | Building 4–9 pm peak vs baseline | EV sessions after 3 pm vs baseline |
|---|---|---|---|
| 2016-06-20 (Mon) | 23.3 vs 21.3 | 728 vs 720 kW | 13 vs 11.9 |
| 2016-07-27 (Wed) | 24.4 vs 21.8 | 769 vs 732 kW | 15 vs 14.0 |
| 2016-07-28 (Thu) | 22.2 vs 20.6 | 762 vs 731 kW | 13 vs 12.0 |
| 2017-06-20 (Tue) | 24.4 vs 20.3 | **775 vs 704 kW (+10%)** | **18 vs 12.4** |
| **2017-06-21 (Wed)** | 25.0 vs 21.0 | **799 vs 728 kW (+10%)** | 12 vs 13.6 |
| 2017-09-01 (Fri) | 38.3 vs 20.8 | 752 vs 736 kW | 10 vs 10.5 |

On average across the six days: building 4–9 pm peak +3 to +10%, EV after 3 pm **+9%**. Both rise in the most expensive window.
**Demo replay days: Tue 2017-06-20 and Wed 2017-06-21.** They're real, in the test period, in summer, and on weekdays, so the $30.18/kW peak-period charge applies.

## 3. Preprocessing pipeline (outputs in `data/processed/`, one script per step)

**Status: implemented** in `headroom/data/` (`python -m headroom.data.pipeline`, about 5 s; invariants in `tests/test_data.py`). Results and every count are in `data/processed/REPORT.md`. Findings from the run:
- Proper timezone handling made the 6 "end ≤ start" sessions from the naive audit disappear: they were **daylight-saving artefacts**, not bad data.
- 3 sessions overlapped on the same port (all on 2016-08-15; a session not closed in the log), so departures were cut to the next arrival. A hard check now guarantees ≤ 6 cars plugged in and EV kW ≤ 6 × 6.2.
- **28.2% of all plugged-in bay time is idle** (the car is full but still occupying the bay). This is the measured basis for the blocked-bay feature.
- BDG2 timestamps were confirmed as local wall-clock (identical weekday morning ramp in summer and winter, 78 buildings).
- The 15-min building series reproduces every hourly value exactly (max error 1.6e-9 kW). 3 days with gaps > 3 h are flagged `imputed_day`.
- AR(1) intra-hour noise (C4) is a **simulator parameter**, not baked into the files, so preprocessing stays deterministic.

Outputs: `sessions.parquet` (18,879 sessions), `ev_15min.parquet`, `building_15min.parquet`, `site_15min.parquet` (70,176 rows; train 366 d, val 120 d, test 245 d).

### A. EV sessions → `sessions.parquet`
1. **Parse dates**: mixed text formats, plus 40 End Date values that are Excel serial numbers (helper in DATA.md).
2. **Timezone**: Start and End are local wall-clock. Use the `Start Time Zone` column (PDT/PST; 6 rows say UTC and are converted from UTC) to resolve DST ambiguity. Store both UTC and America/Los_Angeles.
3. **Filter**: WEBSTER, 2016-01-01 ≤ start < 2018-01-01.
4. **Clean**: drop end ≤ start (6 rows), dwell > 24 h (0.07%), and charging time > dwell + 6 min. Sessions under 0.1 kWh (0.47%, failed plug-ins) are dropped from energy demand but **kept for occupancy**, since they did occupy a bay.
5. **Derive**: port_id (station#port), arrival, departure, dwell_h, charging_h, **idle_h = dwell − charging** (p50 0.02 h, **p90 2.15 h**: the measured "finished EV left in the bay" behaviour behind the blocked-bay feature), kwh, avg_kw.
6. **Assumptions**: requested energy = delivered kWh; deadline = actual departure (standard when driver inputs aren't recorded). Port cap is 6.2 kW.

### B. EV load → `ev_15min.parquet`
Per 15-minute interval: plugged-in count, idle-occupied count, and **uncontrolled power** (each session draws its measured average kW from arrival until its kWh is delivered). Check: total energy equals the sum of session kWh.

### C. Building → `building_15min.parquet`
1. BDG2 timestamps are local wall-clock with 24 rows on DST days (verified), so localise with explicit DST handling: shift the non-existent spring-forward hour and interpolate the repeated fall-back hour.
2. **Gaps**: ≤ 3 h are interpolated linearly; longer gaps flag the day `imputed_day`, and those days are excluded from all metrics.
3. **Hourly → 15-min**: mean-preserving interpolation (PCHIP on cumulative energy, then differenced), so each hour's energy is preserved exactly.
4. **Unobservable intra-hour variation**: headline runs use the real series as is (σ = 0). AR(1) noise with σ ∈ {2%, 5%} is a **labelled stress test** only, never a headline number.

### D. Site assembly → `site_15min.parquet`
building + EV + weather + calendar (hour, day of week, holiday, flex_alert, CPAU season with summer = May 1–Oct 31, ToU period).
**Service limit = 1,000 kW** (assumed 1,000 kVA transformer at power factor ≈ 1; this is also the E-4 TOU upper bound). It's a parameter, so we also report 900 and 1,100 kW.

### E. Scale-up scenarios (hosting capacity)
N-port sites re-use **real recorded Webster port-days**: the date's own 6 ports come first (so N = 6 is the real garage), and each extra port takes a real port-day from the **same day type (workday or not) within ±7 days**, shifted to the target date at the same local clock time. Drawing from neighbouring days (not only the same date) keeps a 36-port fleet from being six copies of the same six cars. Overlapping stays on a virtual port are cut at the next arrival. N ∈ {6, 12, 18, 24, 36, …}. Implemented in `headroom/sim/scenario.py`.

### F. Splits (time-ordered, no leakage)
| Split | Period | Purpose |
|---|---|---|
| Train | 2016-01-01 → 2016-12-31 | Fit models |
| Validation | 2017-01-01 → 2017-04-30 | Tuning, conformal calibration |
| **Test** | **2017-05-01 → 2017-12-31** | All summer demand-charge months, 3 Flex Alert days, and the late-2017 demand drop. **Untouched until the final benchmark on day 9** |

Within test: walk-forward, refitting monthly on all earlier data (as it would be deployed).

### G. Leakage rules
- Web search is **off** in the backtest; its role is played by the frozen event calendar.
- Every feature at time t uses data ≤ t only.
- **No future weather in backtests**: forecasts use only data measured before the issue time (load, temperature lags, calendar). Real weather *forecasts* for 2016–2017 aren't archived, and adding made-up forecast noise would be synthetic. Live mode uses the real Open-Meteo forecast.

## 4. Known issues (all stated in the README)
- **Distribution shift**: Webster sessions fall about 30% from 2017 Q2 to Q4 (stations #2 and #3 most). This is real, falls in the test period, and the forecaster must adapt. We report it.
- The building data is hourly, so 15-min building load is derived (C3–C4).
- **Virtual site**: the building is in Berkeley and the garage in Palo Alto (about 50 km apart, same climate region).
- 2026 tariff applied to 2016–2017 behaviour ("today's prices, historical behaviour").
- Requested energy = delivered energy.
- At today's 6 ports, EV load (about 37 kW) is small next to the building (850 kW peak). **That's the point**: the question is how far the site can scale.

## 5. Real-time inputs (live mode, demo only)
| Feed | Endpoint (verified working) | Use |
|---|---|---|
| Weather forecast | `api.open-meteo.com/v1/forecast` (Palo Alto) | Tomorrow's building-load forecast |
| California grid demand | CAISO Today's Outlook `www.caiso.com/outlook/current/demand.csv` (5-min) | Grid-stress context in the UI |
| CAISO load forecasts and prices | OASIS API `oasis.caiso.com/oasisapi/SingleZip` (zip) | Day-ahead stress signal |
| Flex Alerts and notices | CAISO site plus Muse Spark `web_search` | The agent reads current notices and tightens the 4–9 pm limits |
| Events | Muse Spark `web_search` | Upcoming local events (shown with citations; no effect is claimed unless measured) |
