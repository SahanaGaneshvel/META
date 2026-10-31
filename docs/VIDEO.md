# Demo video: script and shot list (3:00 hard limit)

The final judges watch this live. Rules: show only what was built during the hackathon; every number on screen
comes from a file in `data/results/` (source marked ▶); real data only. Fill [brackets] from the final run.

**Format:** screen capture of the app (1920×1080, browser zoom 110%) + voice-over + 3 title cards. Burned-in
captions (judges may watch muted). Music low. One idea per scene; cut, don't scroll.

| Time | On screen | Voice-over (~150 words/min) |
|---|---|---|
| **0:00–0:10** Hook | Black card, white text over a load spike: **"This 15 minutes cost $2,572."** | "One fifteen-minute charging spike at six p.m. added two and a half thousand dollars to this building's bill. The electricity in it cost three." |
| **0:10–0:25** Problem | Overview page, "The problem" card | "Malls, offices and depots share one meter between the building and the chargers, and the utility bills the highest fifteen minutes of the month. So owners stop adding chargers, or wait years for a grid upgrade." |
| **0:25–0:40** What it is | Overview: the four "How it works" cards, one highlighted at a time | "Headroom sees the building, plans the charging in the room it leaves, and a safety layer guarantees the limit. Meta's Muse Spark runs the site." |
| **0:40–1:10** Heat-wave day | Live replay, 21 June 2017; drag 2 pm → 9 pm; orange climbs into the peak, green stays under it; metric **−$1,066 vs standard smart charging** (month to date at 6 pm; it updates as you drag) | "This is a real heat-wave alert day in Palo Alto, with four times today's chargers. Standard smart charging can't see the building, so it charges straight into the evening peak. Headroom fills the valleys. Same cars, the energy delivered, thirty-two percent lower demand charges." ▶ final_summary |
| **1:10–1:35** Operator by voice | Ask the operator: hold the mic, say *"Fleet vans need forty kilowatt-hours each by six a.m. Can we do it?"*; transcript appears; answer + "every number traced to a tool result ✅"; open "How it got this" | "The site manager just asks. Muse Voice Transcribe hears it, Muse Spark calls the planner. It can only act through checked tools, and every number it says is traced back to the math." |
| **1:35–1:50** Grid alert | Recorded question 2: the alert, sized from past alert days, re-plan result | "On a grid alert it sizes the change from this building's own past alert days, cites the source, and the planner decides." |
| **1:50–2:05** Internet cut | Sidebar toggle **Cut the site's internet** → ask again → *"— Muse Glimmer (on site)"* | "Cut the internet. Muse Glimmer, Meta's open-weight model on the site's own GPU, takes over. The chargers never depended on either." |
| **2:05–2:30** Any building | Check any building: **"Tested on 49 real buildings at 7 US sites: median +15% chargers"**; click *Try a sample building* → answer card | "Does it work beyond one site? On forty-nine real buildings across seven US sites, it fits more chargers in three out of four, a median fifteen percent more, on the same connection. Upload your own meter data and get your answer." ▶ portfolio_summary |
| **2:30–2:45** Checked AI | How the AI is checked: tariff **25/25 fields, bill reproduced** → misread demo **rejected** | "Muse Spark reads the utility's tariff, and we only accept it if it reproduces the real bill. Misread one number and it's rejected." |
| **2:45–3:00** Close | Card: **"48 chargers instead of 36. Same connection. 0 violations."** · Meta: Muse Spark · Muse Voice Transcribe · Muse Glimmer · Built Oct 26–Nov 4 | "The model proposes, the math decides, the safety layer guarantees. Headroom: more chargers, same connection." |

## Shots to capture (in order)
1. Overview (static, 5 s). 2. Live replay drag 14:00 → 21:00 at 2 s per step (speed up in edit).
3. Operator: voice request (Live mode; costs < $0.01), then recorded question 2 (Replay).
4. Sidebar toggle → Glimmer answer (Replay). 5. Check any building: sample. 6. Tariff verify + misread.
7. Title cards (`docs/cards/`).

## Before recording
- [ ] Numbers in the cards match `data/results/final_summary.csv`, `capacity_summary_test.csv`, `portfolio_summary.csv`
- [ ] Glimmer recordings present (docs/GLIMMER_KAGGLE.md); app in Replay except the voice shot
- [ ] Browser: clean profile, no bookmarks bar, 110% zoom; notifications off
