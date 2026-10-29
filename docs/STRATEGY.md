# Headroom — strategy to win the Indie track

**Market: United States first**; India is future scope (section 9). Revised 2026-10-29 after reading the full participant guide and losing access to the Meta Model API (no usable card): every model job now runs on **Meta's open-weight models** (Muse Spark Open, Muse Glimmer) through partner inference.

## 0. How the competition is judged (what we optimise for)

- Stage 1 pass/fail: fits a problem statement and uses Meta models in work built during the hackathon. Ours: **Push the Frontier → energy → control a physical system.**
- Stage 2: each criterion 0–10. **Impact 30 · Meta tooling 25 · Demo 25 · Technical depth 20.**
- Stage 3: top 5 per track; final judges **watch the 3-minute videos live** (Nov 9) and re-rank. The video decides first place.
- Judges use the video, the code **and the live product**, so the hosted demo must keep working Nov 5–8 even if credits run out.
- Only work built during the hackathon may be shown, and it must be clearly identifiable. Repo history starts 2026-10-27 (after kickoff). The IndiaAI notebook is prior work: disclosed, never demoed.
- Bonus prizes we qualify for by design: **Best Build on Meta Open Weight Models ($10K)**, **Best Agentic Use-Case ($5K)**, Most Viral on X ($20K), Community Vote ($10K).

## 1. What a sharp judge would find in the original idea, and the fix

| # | Hole | Fix |
|---|---|---|
| 1 | "Zero violations vs. a static cap" is solved: we measured that ACN-Sim's LLF **and** Caltech's adacharge MPC both get 0 violations out of the box (`scripts/baselines_check.py`). | Benchmark against **static cap, LLF and adacharge MPC**; put the novelty where they are blind (row 2). |
| 2 | The forecast's value isn't pinned down. | EV schedulers assume a *fixed* EV capacity. Real sites share one meter with an *uncertain* building load, and the bill depends on the combined 15-minute peak. Headline = **probabilistic hosting capacity**: "this site can safely host N more chargers at <1% risk of setting a new demand peak." |
| 3 | The model could look like a chat wrapper. | Every model job has a **verifier and a number** (section 4). "The model proposes, the math decides, the safety layer guarantees." |
| 4 | Queue counting from a camera is a weak signal. | Use video for what meters can't see: **bays occupied but not charging** (28.2% of plugged-in time is idle: measured). |
| 5 | Mixed India/US story. | One coherent US site, same years, every choice backed by an audit (`docs/DATA_SPEC.md`). |
| 6 | A "match-night surge" isn't real here: Stanford game days showed **no effect (0.97×)**. | Replay real **CAISO Flex Alert** heat days instead (building 4–9 pm peak **+10%**, measured). |
| 7 | "Ages the transformer" has no model. | IEEE C57.91 thermal model → **loss of life (equivalent hours)**. |
| 8 | Prior forecasting code can't be shown as new work. | **Rebuild** the forecaster during the hackathon (same proven approach, new code, plus calibrated quantiles); disclose the approach in the README. |

## 2. The US problem in one number

Palo Alto's CPAU **E-4 TOU** (effective 2026-07-01, `data/docs/`) bills two demand charges on the highest **15-minute** average kW of the month. Summer: **$30.18/kW** peak-period (4–9 pm weekdays) + **$21.27/kW** max demand (any time). A single +50 kW spike for 15 minutes at 6 pm on a summer weekday adds 50 × $51.45 = **$2,572** to that month's bill; the 12.5 kWh it delivers costs $2.69. **The spike costs about 950× its energy.** That's why owners stop adding chargers, and it's the video's opening line.

## 3. Pitch (use this wording everywhere)

> **Headroom lets a site add EV chargers on the grid connection it already has, without a service upgrade and without ever setting a new demand-charge peak, and tells the owner in advance exactly how many it can safely add.**
> A probabilistic forecaster sizes the headroom, a verified controller enforces the limits, and Muse Spark Open runs the site as its operator: it reads the utility's tariff, takes requests by voice, watches the bays, and answers "can I add twelve more chargers?" from simulation, not guesswork. If the network drops, Muse Glimmer keeps the site running.

Headline numbers (hypotheses until measured on the test split):
- **Additional chargers hosted** on the same service at <1% risk of a new monthly peak, vs. static cap, LLF and adacharge MPC.
- **Demand-charge dollars saved per month** and total bill under E-4 TOU.
- **Violations: 0** in 10,000 fuzzed plans, including adversarial model outputs.
- Energy delivered / requested, transformer loss of life (h), peak 15-min kW.
- Agent accuracy: tariff fields correct (%), bill reproduced within ±1%, blocked-bay precision/recall, fine-tuned Glimmer before/after.

## 4. Meta models: every job is load-bearing and checked

All calls go through one OpenAI-compatible client; provider (Baseten, Fireworks, Modal, Groq) is configuration. Every model response used in the demo is **recorded and replayable**, so the hosted demo survives quota loss.

| Model | Job | Verifier |
|---|---|---|
| **Muse Spark Open** (tool calling, strict JSON) | Operator agent: plain language or voice → `set_requirement`, `replan`, `simulate_expansion`, `explain_plan`, `read_tariff` | Every tool call is schema-checked; the safety layer re-checks every plan; fuzz test with adversarial outputs |
| **Muse Spark Open** (text + page images, 1M context) | Tariff document → strict `TariffSpec` JSON (seasons, ToU windows, both demand charges, 15-min interval, holiday rule) | **Reproduces a month's bill** from metered data within ±1%; field accuracy against hand-checked gold; second tariff (E-7 TOU, then PG&E's subscription-style EV tariff) to show it generalises |
| **Muse Spark Open** (tools: CAISO alerts, live weather) | Grid-stress events → proposed tighter 4–9 pm ceiling or demand multiplier, with citations | Accepted only inside the forecast's calibrated bounds; backtested on the real 2016–2017 Flex Alert days with a frozen calendar (no leakage) |
| **Muse Spark Open** (video input) | Bay camera → "occupied but not charging" bays | Cross-checked against charger session data; precision/recall on labelled clips |
| **Muse Spark Open** (audio input) | The site manager speaks a request; no separate speech model | Same tool validation as typed requests |
| *(optional)* **Meta MMS-TTS** (open weights) | Spoken replies, only if time allows; default is the browser's built-in voice, which isn't a model call | — |
| **Muse Glimmer** | Offline backup operator ("pull the network, the site stays safe") | Same tools and validators |
| **Muse Code** | Coding agent during the build | Mentioned in README and video |

**Runtime rule (from Meta):** every model the project calls at runtime must be a Meta model; open Meta models count however they're run. So **no TinyTimeMixer, Whisper or third-party voices**. Models we train ourselves on our own data (LightGBM quantile forecaster) are our code, not third-party models. The README lists every Meta model used and where.

## 5. Architecture

```
 Tariff docs ─┐                       ┌─ Bay camera (MP4) ─► Spark Open video ─► blocked bays
 Grid alerts ─┤                       │
 Operator ────┼─► Spark Open agent ───┤  typed, validated tools        (Glimmer if offline)
 (voice/text) ┘   (proposes)          │
                                      ▼
 History ─► Probabilistic forecaster ─► MPC planner (LP, HiGHS, 15-min, ceiling on the
            (hour-of-day + LightGBM        │          building's calibrated quantile, ToU cost)
             quantiles, conformal)         ▼
                                  DETERMINISTIC SAFETY LAYER (last measured load + ramp margin,
                                  service kW limit, demand ceilings, port caps)
                                           │  nothing above can override
                                           ▼
                        OCPP 2.0.1 SetChargingProfile ─► simulated chargers
```

## 6. Demo storyboard (3-minute video; the hosted demo follows the same script)

1. **0:00–0:15 Hook.** "This 15-minute spike cost $2,572." One meter, 24 chargers wanted, the service is full on paper.
2. **0:15–0:45 The agent reads the tariff.** Drop in the real CPAU PDF; Spark Open extracts both demand charges and the windows; the bill reproduces within 1%.
3. **0:45–1:35 Real heat-wave day (Wed 21 June 2017, CAISO Flex Alert).** Three controllers side by side at 4× today's chargers, with a live bill counter: static cap leaves cars uncharged; adacharge MPC, blind to the building, sets a new monthly peak; Headroom stays under it. The camera flags a blocked bay; the agent re-plans and explains.
4. **1:35–2:00 Voice request.** "Fleet vans need 80% by 6 am." The agent accepts with a probability, or pushes back.
5. **2:00–2:20 Pull the network.** Glimmer takes over; the site stays safe.
6. **2:20–2:45 "Can I add 12 more chargers?"** "Yes to 8 at 0.4% risk; 12 needs a battery or an upgrade."
7. **2:45–3:00 Proof.** Benchmark table, 10,000 fuzzed plans with 0 violations, "built during the hackathon" card, future scope (any utility; India next).

## 7. Scope

**Core (by the night of Nov 1; scores ≈8.5 alone):** site simulator (building + EV + service limit + 15-min demand + transformer) · baselines (static cap, LLF, adacharge MPC) · probabilistic forecaster with backtest · MPC + safety layer + fuzz tests · Spark Open agent with tools and record/replay · tariff extraction + bill check · hosting-capacity Monte Carlo · control-room UI.
**Priority (revised Oct 29): one complete path on screen first** (forecast → tariff verified → blocked bay → savings), then the video. **Extras after that, in order; whatever isn't done by end of Nov 3 is cut:** hosted demo with replay → ablations + intra-hour stress test → voice → Glimmer offline. **Cut:** Glimmer fine-tune, OCPP output, ACN-Data (research-only licence), PG&E/URDB tariffs (optional generalisation check only).
**Won't:** real hardware, mobile app, multi-site, India site, driver payments.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Partner credits arrive late or not at all | All model code runs against any OpenAI-compatible provider; develop against whichever key arrives first (Groq/Fireworks/Baseten); recorded responses keep the demo working |
| API outage or quota during judging | Hosted demo defaults to replaying recorded responses; "go live" switch |
| Headroom doesn't beat adacharge MPC on a metric | Headline is hosting capacity under *uncertain building load*, which MPC doesn't model. Report honestly either way |
| Laptop (8 GB, no GPU) too slow | LP planner on HiGHS; Monte Carlo capped; Modal for heavy jobs |
| Running out of time | Core first; extras gated; feature freeze Nov 3 night; video Nov 4 |
| 2026 tariff on 2016–2017 behaviour | Stated plainly: "today's prices, historical behaviour" |

## 9. Future scope (README + one line in the video)

- **India**: the agent reads the tariff, so the architecture is utility-agnostic. Next: Tamil Nadu (TNERC EV time-of-day slots, kVA sanctioned-load limits, Chennai malls and fleet depots).
- Other US utilities (PG&E BEV, SCE TOU-EV): same agent, different document.
- Batteries and solar as extra controllable assets; real OCPP hardware pilots.
