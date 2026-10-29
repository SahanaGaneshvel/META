# Demo video — script and shot list (3:00 hard limit)

Judges watch this live in the final round. Rules: show only what was built during the hackathon; every number on screen comes from a script in `scripts/` (marked ▶ below with its source); real data only. Numbers in [brackets] are filled from the final benchmark on the held-out months before recording.

**Format:** screen capture of the control room (1920×1080) + voice-over + three title cards. Captions burned in (judges may watch muted). Music low.

| Time | On screen | Voice-over (≈150 words/min) |
|---|---|---|
| **0:00–0:12** Hook | Black card, white text, a 15-minute spike on a load chart: **"This 15 minutes cost $2,572."** | "One fifteen-minute spike at six p.m. cost this building two thousand five hundred dollars. The electricity in it cost less than three." |
| **0:12–0:30** Problem | Control room, building load line; caption: *one meter · building + chargers · 1,000 kW service* | "Malls, offices and depots share one meter between the building and the chargers. The utility bills the highest fifteen minutes of the month. So owners stop adding chargers, or cap them and waste the room they have." |
| **0:30–0:42** What Headroom is | Architecture card (one line each): forecaster → planner → safety layer; Muse Spark Open operator; Muse Glimmer backup | "Headroom runs the chargers inside the room the building leaves. A calibrated forecast sizes it, a planner fills it, a safety layer guarantees the limit, and Muse Spark Open runs the site." |
| **0:42–1:05** Tariff, verified | Tariff tab: PDF → JSON → **"bill reproduced within [x]%"** ▶ `read_tariff`; then the misread demo: **"24/25 fields right → bill off 13.8% → rejected"** | "First, Spark Open reads the utility's actual tariff. We don't trust it: we bill two years of real meter data with what it read. It matches. Read one number wrong and the bill is off fourteen percent, so it's rejected." |
| **1:05–1:45** The real heat-wave day | Replay **21 June 2017 (real CAISO Flex Alert)**, 24 chargers. Slider sweeps 2 pm → 9 pm. Three lines: static cap, standard smart charging, Headroom; forecast band; live counter **"demand charges added: $2,670 vs $4,231 per month (test months)"** ▶ `final_benchmark` | "This is a real heat-wave day in Palo Alto, replayed at four times today's chargers. The standard smart charger can't see the building, so it stacks cars onto the peak. Headroom sees the building's forecast and fills the valleys. Same cars, nearly all the energy delivered, thirty-seven percent lower demand charges." |
| **1:45–2:05** Operator by voice | Operator tab; speak: *"Fleet vans need forty kilowatt-hours each by six a.m. Can we do it?"* Trace shows `set_requirement` ✅ with the probability | "The site manager just asks. Spark Open calls the planner. Every tool call is checked before it runs; the answer comes with a probability from the calibrated forecast, not a guess." |
| **2:05–2:20** Blocked bay | Bays tab: annotated clip, **bay N blocked** (camera sees a car, charger reports no power) | "Meters can't see a bay that's occupied but not charging. Spark Open watches the camera; the charger data confirms it. In the real data, that is 28 percent of plugged-in time." |
| **2:20–2:35** Can I add chargers? | Expansion tab curve: chargers vs. added demand charges, three controllers; agent answer for +18 ▶ `capacity_study test` | "Ask whether to expand, and it answers from simulation on real sessions: how much it costs, how often drivers are fully served, and zero limit violations." |
| **2:35–2:48** Proof | Proof tab: held-out benchmark + ablations (oracle bound: Headroom gets 69% of what is physically possible), P99 coverage [99.6]%, **10,000 fuzzed plans · 0 violations** | "On months it never saw. Within reach of a perfect-foresight controller. Ten thousand adversarial plans, zero violations." |
| **2:48–3:00** Close | Card: **"Built during the hackathon, Oct 26–Nov 4: everything shown."** Meta models: Muse Spark Open · Muse Glimmer. Real data: Palo Alto · UC Berkeley · CPAU · CAISO. Next: any utility, India. | "The model proposes, the math decides, the safety layer guarantees. Headroom: more chargers, same connection." |

## Shots to capture (in order of recording)
1. Tariff tab: live extraction result + misread expander.
2. Replay: slider 14:00 → 21:00 on 2017-06-21, steady 2 s per step (screen recorder at 60 fps; speed up in edit).
3. Operator: voice request (record the mic audio separately as backup), expand the tool trace.
4. Bays tab: annotated still + decision table.
5. Expansion tab, Proof tab: slow scroll.
6. Title cards (Canva/Figma; same font as UI).

## Before recording
- [ ] Final numbers filled in from `data/results/final_summary.csv` and `capacity_summary_test.csv`
- [ ] All model calls in the script recorded (`data/recordings/`), app in **Replay recorded** mode so takes are identical
- [ ] Owner's own bay footage swapped in, plates and faces blurred, caption "illustrates the problem"
- [ ] Check every on-screen claim against its source script
