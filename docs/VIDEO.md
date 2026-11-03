# Demo video

The video has no voice. Captions tell the story, burned into the picture, so it works muted. Every
number and screen comes from the real app, the real data or the utility's own tariff document.

## Make it
```powershell
.\.venv\Scripts\python.exe -m pip install -r scripts\video\requirements-video.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe scripts\video\capture.py    # drives the real app, screenshots -> video/shots/
.\.venv\Scripts\python.exe scripts\video\build.py      # -> video/headroom_demo.mp4 (1080p, about 2:30, silent)
```

## The story (about 2:30)
| Scene | On screen | Caption |
|---|---|---|
| Title | Headroom over the real heat-wave day's meter data | (none) |
| The want | Dark card | Offices, malls and depots want more EV chargers. But the chargers share the building's grid connection. |
| The bill | The real Palo Alto E-4 tariff page, demand charges circled | Utilities bill a business for its highest 15 minutes… $30.18 + $21.27 for every kW of that peak. |
| The cost | Card: "This 15 minutes cost $2,572." | (on the card) |
| The cause | The real building's meter on 21 June 2017; standard smart charging stacks onto the evening peak and sets a new monthly peak | Standard smart charging can't see the building… |
| The result today | Card | So owners install fewer chargers, or wait for a grid upgrade. |
| Headroom | Title card + the four steps | (on the card) |
| The day, replayed | The app's Live replay, 2 pm → 9 pm | Headroom charges in the room the building leaves… 32% lower demand charges. |
| The operator | Ask the operator: the recorded conversation | Muse Spark answers through checked tools… every number traced to a tool result. |
| Internet cut | The sidebar toggle; Muse Glimmer answers | Meta's open-weight Muse Glimmer takes over on site. |
| Beyond one building | Check any building: 49 buildings, the sample | More chargers in 77% of them, fewer in none. |
| Trust | PG&E tariff read 21/21; OCPP 480/480 | Read only if it reproduces the real bill; chargers driven over OCPP. |
| Result | Card: 48 chargers instead of 36 | (on the card) |
| Close | Card | (on the card) |

## Finish it in Clipchamp (free, built into Windows 11; about 15 minutes)
1. Open **Clipchamp**, then **Create a new video**, and import `video/headroom_demo.mp4`.
2. **Optional real-world opener (5–8 s).**
   - Search Clipchamp's free **Stock video** for "EV charging station" or "electric car charging parking". Pexels also works and is free to use.
   - Drag a clip to the start of the timeline.
   - Add a **Text** overlay: *"This building wants more EV chargers."*
   - Keep the total under 3:00.
3. **Music.**
   - Open **Content library**, then **Audio**, then **Music**, and pick a calm, upbeat track.
   - Drag it under the whole video and set its volume to about 20%.
   - Fade it out over the last 3 seconds.
4. **Export** at 1080p and check the length is under 3:00.
5. **Upload to YouTube** as *Unlisted* or *Public*.
   - Thumbnail: `docs/cards/thumbnail.png`.
   - Description: use the text below.

## YouTube description
```
Headroom: an AI operator that fits more EV chargers on the grid connection a building already has.

Offices, malls and depots want more EV chargers, but the chargers share the building's grid connection, and
utilities bill the highest 15 minutes of each month. Headroom watches the building, charges the cars in the
room the building leaves, and guarantees the site never goes over its limit. Meta's Muse Spark runs the site:
the manager just asks it questions, and every number in its answers is checked against the math.

Built for the Meta Global AI Developer Hackathon 2026 (Indie track).

WHAT YOU'LL SEE
• The real problem: a real building's meter, a real utility tariff, and what one charging spike costs.
• A real heat-wave day replayed: standard smart charging vs Headroom.
• Muse Spark answering the site manager through checked tools.
• The internet cut: Meta's open-weight Muse Glimmer takes over on site.
• 49 real buildings, a second utility's tariff, and chargers driven over OCPP.

RESULTS (held-out months, real data)
• 48 chargers instead of 36 on the same connection; every driver fully charged; 0 limit violations.
• 32% lower demand charges than standard smart charging.
• More chargers in 77% of 49 real buildings, fewer in none.

META MODELS
Muse Spark 1.3 · Muse Voice Transcribe 1.0 · Muse Glimmer (open weights)

EXPLORE THE PROJECT
Source code: [GitHub link]
Try the app: [app link]

The data is real: Palo Alto charging sessions, Building Data Genome 2 meters, published utility tariffs.
The site that combines them and the chargers are simulated. Research prototype; no utility or Meta
endorsement is implied.
```
