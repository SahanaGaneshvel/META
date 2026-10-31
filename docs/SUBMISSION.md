# Submission text (paste into the form; fill [brackets] from data/results/)

**Project name:** Headroom

**Tagline (one line):** More EV chargers on the grid connection a site already has: an AI operator on Muse Spark, with the math in charge.

**Track / theme:** Indie · Push the Frontier (energy: controlling a physical system)

## What it does
Malls, offices and depots share one meter between the building and its EV chargers, and US utilities bill the highest 15 minutes of each month: one 50 kW spike at 6 pm can add $2,572 to a bill. So owners stop adding chargers or wait years for a grid upgrade.

Headroom forecasts the building's load, plans the charging in the room the building leaves with a linear program re-solved every 15 minutes, and guarantees the connection limit with a safety layer nothing can override. Muse Spark runs the site: it reads the utility's tariff (accepted only if it reproduces the real bills), takes requests by voice (Muse Voice Transcribe) or text, sizes plan changes for grid alerts from the building's own history, spots blocked bays on camera, and answers "can I add 12 more chargers?" from simulation. It can act only through validated tools, and every number in its answers is traced to a tool result. If the internet drops, Muse Glimmer on the site's own GPU takes over; charging never depends on either model.

## Results (held-out months, real data)
- [48] chargers instead of [36] on the same 1,000 kW connection, every driver served, 0 limit violations.
- 37% lower demand charges than standard smart charging and Caltech's open-source adacharge MPC ($2,670 vs $4,231/month at 24 chargers), 99.8% of energy delivered; 69% of the saving a perfect-foresight controller gets.
- Across [N] real metered buildings at [S] US sites: a median [+x]% more chargers on the same connection.

## How we used Meta's models
- **Muse Spark 1.3** (Meta Model API): tool-calling site operator; tariff document → structured rates, verified by bill reproduction (25/25 fields, 0.0% bill error); bay video → occupancy, cross-checked with charger power (6/6 bays).
- **Muse Voice Transcribe 1.0** (Meta Model API): spoken requests → text.
- **Muse Glimmer** (open weights, on-site GPU): automatic failover operator when the cloud is unreachable.
Every output has a verifier; every call is recorded with its provenance and replayed in the demo.

## How we built it
Python: a 15-minute simulator on real Palo Alto charging sessions and BDG2 building meters; the planner (SciPy HiGHS); the safety layer (fuzz-tested on 10,000 plans); a calibrated hour-of-day forecast; pydantic tool schemas; a provider-agnostic Meta model client with record/replay, failover and a hard spend guard; a Streamlit app. Everything was written during the hackathon; Caltech's adacharge/acnportal are used unmodified as a baseline.

## Challenges
Driving charging into the building's valleys without starving drivers (the strict policy serves only 86%, so we made drivers-first the default); keeping an LLM out of the control loop while making it genuinely useful; proving the gain beyond one site.

## What's next
Pilots on real chargers over OCPP; any utility's tariff (the reader is document-driven); India (TNERC time-of-day EV tariffs, kVA-limited malls and fleet depots); batteries and solar as extra controllable assets.

**Links:** repo [URL] · live demo [URL] · video [URL]

---

## X post (by Nov 8, for the viral prize)
> One 15-minute EV charging spike can add $2,572 to a building's power bill. ⚡
>
> We built Headroom for the @Meta AI hackathon: Muse Spark runs the site, the math decides, a safety layer guarantees the grid limit.
>
> [48] chargers instead of [36] on the same connection. 0 violations. Real data.
>
> [video link] #MetaAIHackathon
