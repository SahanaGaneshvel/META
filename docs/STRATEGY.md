# Headroom — strategy to win the Indie track

**Market: United States first**; India is future scope (section 9).

## 1. Honest verdict on the application idea

The idea is strong: it controls a real physical system, you have a proven forecasting record, it can be measured, and it puts hard safety limits under an LLM. That is a top-tier foundation. As written, though, a sharp judge would find five holes. Each one costs points under the actual judging weights (Impact 30, Meta tooling 25, Demo 25, Technical depth 20).

| # | Hole a judge will find | Why it costs points | Fix |
|---|---|---|---|
| 1 | **The baseline is too easy to beat.** We benchmark against static caps. | We measured this (`scripts/baselines_check.py`): ACN-Sim's least-laxity-first scheduler **and** Caltech's MPC (`adacharge.AdaptiveSchedulingAlgorithm`) both get **0 violations** (peaks of 147 and 150 kW vs 213 kW uncontrolled), out of the box. "Zero violations vs. a static cap" is a solved problem. | Benchmark against the **strongest open baselines**: static cap, LLF, and adacharge MPC. Put the novelty where they are blind (row 2). |
| 2 | **The forecast's value isn't pinned down.** | ACN-style schedulers assume a *fixed* EV capacity. Real sites share one meter with an *uncertain* building load, and the bill depends on that combined 15-minute peak. The forecast matters for three decisions those schedulers can't make: (a) **promises**: can I guarantee the fleet 80% by 6 am? (b) **admission**: can I accept this booking? (c) **sizing**: how many chargers can I add? | Make **probabilistic hosting capacity** the headline: "this service can safely host N more chargers at <1% risk of setting a new demand peak." It's one number, it's what the owner actually pays for, and only a probabilistic forecast can produce it. |
| 3 | **Muse Spark could look like a chat wrapper.** "Use of Meta tooling" (25%) rewards going *beyond basic integration*. | Tool-calling chat is the default integration most entrants will submit. | Give every model job a **verifier** and a number: tariff PDF → strict-schema `TariffSpec` → **checked by reproducing a bill**; event search → forecast adjustment **accepted only within ensemble bounds**; camera → SAM 3.1 tracks + Spark video reasoning → **blocked-bay detection**. Report each job's accuracy. |
| 4 | **The camera's "arrival signal" is weak.** | Most sites don't have queues; the forecast already predicts arrivals. | Use the camera for what meters *can't* see: **bays occupied but not charging** (a petrol car in an EV bay, or a finished EV left idle). SAM 3.1 tracks vehicles inside bay polygons, the charger reports no session → "bay 4 blocked 38 min." The controller removes the bay from the plan. |
| 5 | **The data didn't match the story** (Indian rules, US data). | Judges notice mixed sources. | **Solved by going US-first.** Bay Area drivers (Palo Alto WEBSTER) + Bay Area building (Berkeley) + Palo Alto tariff + California grid events, same years (2016–2017). Every choice is backed by a measured audit: see `docs/DATA_SPEC.md`. |
| 6 | **Search leaks the future.** | Meta `web_search` has **no date filter** (checked in the docs). Searching while replaying a past day lets the model read the outcome, which invalidates the evaluation. | Two modes. **Backtest**: a frozen, cited calendar of *real* 2016–2017 events (CAISO Flex Alert grid-emergency days, federal holidays); no live search. **Live**: real-time weather (Open-Meteo), California grid state (CAISO), and current Flex Alerts/notices via search, with citations. (Stanford game days were **tested and rejected**: no effect on charging, 0.97×; see DATA_SPEC.) |
| 7 | **"Ages the transformer" is a claim without a model.** | Technical depth (20%) rewards real craft. | Add the IEEE C57.91 thermal model (top-oil → hot-spot → loss of life). Report **transformer ageing (equivalent hours)**. About 60 lines; few entrants will have it. |
| 8 | **Scope is too wide for one person in 10 days.** | A half-finished system loses to a polished narrow one. | Strict cut-list (section 6). Feature-freeze at the end of day 7. |

**Can this win first place?** No plan can promise that. With these fixes it scores strongly on every criterion, and it has something most entries won't: a physical system, a hard safety guarantee, and real numbers. That is how you get into the top five. Execution and the demo video decide the order within it.

## 2. The US problem in one number

Palo Alto's medium-commercial time-of-use tariff (CPAU **E-4 TOU**, effective 2026-07-01, in `data/docs/`) bills **two demand charges** on the single highest **15-minute** average kW of the month:

| Summer (May–Oct) | Rate |
|---|---|
| Peak-period demand (4–9 pm weekdays) | **$30.18/kW** |
| Max demand (any time) | **$21.27/kW** |
| Peak energy | $0.21556/kWh |

A single unmanaged 15-minute spike of +50 kW at 6 pm on a summer weekday adds 50 × ($30.18 + $21.27) = **$2,572 to that month's bill**. The energy in that spike (12.5 kWh) costs $2.69. **The spike costs about 950× the energy it delivers.** That is why US site owners stop adding chargers, and it's the line to open the video with.

Two details the agent should find in the PDF, not hard-coded:
- The demand window is **15 minutes**, the same as our control step, so the optimiser targets exactly what the meter bills.
- "If the Customer's load is intermittent or subject to fluctuations, the City may use a **5-minute** interval" (E-4 sheet). EV load fluctuates, so the agent should flag this risk.

## 3. Revised pitch (use this wording everywhere)

> **Headroom finds the spare capacity in the grid connection a site already has, and runs EV chargers inside it, without ever setting a new demand peak.**
> A probabilistic forecaster sizes the headroom, a verified controller enforces the limits, and Muse Spark acts as the site's operator: it reads the utility's tariff, watches for demand shocks, takes requests in plain language, and answers "can I add six more chargers?" from simulation, not guesswork.

Headline numbers to produce (targets are hypotheses until measured):
- **Additional chargers hosted on the same service** at <1% risk of a new monthly peak, vs. static cap, LLF and adacharge MPC.
- **Demand-charge dollars saved per month** and **total bill ($)** under CPAU E-4 TOU.
- **Violations: 0** in 10,000 fuzzed plans, including adversarial LLM outputs.
- **Energy delivered / energy requested**, **transformer loss of life (h)**, **peak 15-min kW**.
- **Agent accuracy**: tariff fields extracted correctly (%), bill reproduced within ±1%, blocked-bay detection precision/recall.

## 4. What each Meta model does (each job is load-bearing and checked)

| Model | Job | How it's verified |
|---|---|---|
| **Muse Spark 1.3** (PDF input + structured output) | CPAU tariff sheet → strict JSON `TariffSpec` (seasons, ToU windows, two demand charges, 15-min interval, holiday rule, 5-min clause) | Recomputes a bill from metered data; field-level accuracy against a hand-labelled gold file; tested on E-4 TOU **and** E-7 TOU (does it generalise?) |
| **Muse Spark** (tool calling) | Operator agent: plain language → `set_requirement`, `replan`, `simulate_expansion`, `explain_plan` | Every tool call is type-checked; the safety layer re-checks every plan |
| **Muse Spark** (`web_search`) | Live mode: current CAISO Flex Alerts, heat-wave warnings and grid notices → proposed tighter 4–9 pm limits or demand multiplier, with citations | Accepted only inside forecast-ensemble bounds; backtested on the 6 real 2016–2017 Flex Alert days |
| **SAM 3.1** (video, text prompt "car") | Tracks vehicles in bay polygons → occupied-but-not-charging detection | Precision/recall on a small labelled clip set |
| **Muse Spark** (video understanding) | Second opinion on flagged bays ("petrol sedan, no cable") | Shown to the operator only, never used for control |
| **Muse Code** | Our coding agent during the build | Mentioned in the README and video |
| **Open weights on Baseten/Modal** (optional) | Offline fallback agent | Demo: pull the network, and the controller stays safe |

Design rule we repeat in the video: **the LLM proposes, the math decides, the safety layer guarantees.**

## 5. Architecture

```
 Tariff PDFs ─┐                      ┌─ Camera (MP4) ─► SAM 3.1 ─► bay occupancy/blocked
 Notices ─────┤                      │
 Operator ────┼─► Muse Spark agent ──┤  tools (typed, validated)
 Web search ──┘   (proposes)         │
                                     ▼
 History ─► Probabilistic forecaster ─► MPC optimiser (cvxpy/HiGHS, 15-min, P90 headroom,
            (HoD + TTM + LightGBM,          │            two demand charges + ToU energy)
             stacked, conformal quantiles)  ▼
                                   DETERMINISTIC SAFETY LAYER  (measured load feedback,
                                   service kW limit, 15-min demand ceiling, transformer hot-spot)
                                            │  nothing above can override
                                            ▼
                         OCPP 2.0.1 SetChargingProfile ─► simulated chargers (ACN-Sim)
```

## 6. Demo storyboard (≈3-minute video; the live demo follows the same script)

1. **0:00–0:20 The 950× problem.** A mixed-use building in downtown Palo Alto: one meter, 12 chargers in the garage, owner wants 24. One 15-minute spike costs $2,572; a service upgrade takes months and costs a lot.
2. **0:20–0:50 The agent reads the tariff.** Drop in the real CPAU E-4 TOU PDF. The agent extracts both demand charges, the ToU windows and the 15-minute rule, flags the 5-minute clause, and reproduces a month's bill to within 1%.
3. **0:50–1:40 Heat-wave grid emergency (replay of real Wed 21 June 2017, a CAISO Flex Alert day).** The building's 4–9 pm peak really rose +10% (799 vs 728 kW) and EV demand after 3 pm rose about 9%, both in the $30.18/kW window. Split screen of three controllers (static cap, adacharge MPC, Headroom) at 4× today's charger count. The agent knows the Flex Alert, the camera flags two blocked bays, and the agent re-plans and explains. Static cap leaves cars uncharged; MPC, blind to the building load, sets a new monthly peak; Headroom stays under it.
   Then a **live cut**: tonight's real Palo Alto weather forecast and the California grid's demand right now.
4. **1:40–2:10 "Fleet vans need 80% by 6 am."** Plain language in; the agent accepts with a probability, or pushes back ("72% achievable at P90; here's why").
5. **2:10–2:40 "Can I add 12 more chargers?"** Monte Carlo hosting capacity: "Yes to 8 at 0.4% risk; 12 needs a 40 kWh battery or a service upgrade."
6. **2:40–3:00 Proof.** Benchmark table, the 10,000-plan fuzz test with 0 violations, the architecture, and a closing line, plus a two-second future-scope note: "any utility tariff, next: India."

## 7. Scope (cut-list)

**Must (days 1–7):** site simulator (building load + EV sessions + service limit + 15-min demand + transformer thermal model) · four baselines · probabilistic forecaster with backtest · MPC + safety layer + fuzz tests · agent with four tools · tariff-PDF extraction + bill check · hosting-capacity simulation · control-room UI.
**Should (days 7–8):** SAM 3.1 blocked-bay detection · live search mode · OCPP output.
**Won't:** real hardware, mobile app, multi-site, India site (future scope), Muse Image, driver payments.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| API quota or outage during the live demo | Cache every agent response from rehearsals; a "replay cached" switch in the UI; record the video by day 9 |
| The $150 of credits runs out | `reasoning_effort` "low" for routine calls; batch tests; avoid `-contributor` models for anything private |
| SAM video format or limits differ from the docs | Spike on day 1; fall back to per-frame image segmentation at 1 fps |
| ~~Game days show no demand effect~~ | **Measured: no effect (0.97×), so dropped.** Replaced by Flex Alert heat days, where a real effect was measured (building +10%, EV after 3 pm +9%) |
| Fixes don't beat adacharge MPC on the headline | Headline = hosting capacity under *uncertain building load*, which MPC doesn't model. Report honestly either way |
| Laptop (8 GB RAM, no GPU) too slow | TTM runs on CPU; cap Monte Carlo at about 500 scenarios; use Modal credits if needed |
| 2026 tariff applied to 2016–2017 load data | State it plainly: "today's tariff, historical behaviour" (the standard practice in bill-impact studies) |

## 9. Future scope (README + one line in the video)

- **India.** The architecture is utility-agnostic because the agent reads the tariff. Next market: Tamil Nadu (TNERC EV time-of-day slots, kVA sanctioned-load limits, Chennai malls and fleet depots), where grid connections are even tighter and EV growth is fastest. This is the founder's own city.
- Other US utilities (PG&E BEV, SCE TOU-EV): the same agent, different PDF.
- Batteries and solar as extra controllable assets; real OCPP hardware pilots.

## 10. Rules

Read the Official Rules for team size, submission format and video length. The IndiaAI forecasting approach is prior work from 2025: reuse it only as the rules allow, and keep it **disclosed in the README**.
