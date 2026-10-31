# Final plan: Headroom to a finished, winning submission

Deadline: Nov 4, 9 PM ET = **Nov 5, 7:30 AM IST**. Target: everything built and verified on **Oct 31**;
Nov 1–2 buffer and owner tasks; video Nov 3; submit Nov 4 evening IST.

Judging: Impact 30 · Meta AI tooling 25 · Demo 25 · Technical depth 20. The top 5 per track are re-ranked
from the **video**, watched live. Bonus prizes in reach: Best Build on Meta Open Weight Models ($10K,
Glimmer), Best Agentic Use-Case ($5K), Most Viral on X ($20K), Community Vote ($10K).

## Budget rule (hard)
- Total Meta API spend stays **under $100**. In code: every call books its worst case ($4.50) before it is
  sent; budget $50 (`.env`), hard ceiling **$95** in `headroom/meta/client.py` that `.env` cannot raise;
  no automatic retries. Running total: `data/spend.json`.
- Expected spend for everything below: **under $3** (re-recordings ~$0.10 each, tariff readings ~$0.02,
  a voice test < $0.01). Simulations, studies and Glimmer (own GPU) cost nothing.
- Live calls only when a step below needs one; spend reported after each.

## A. Close the forecaster question (staff: "profile avoids the question")
- [x] Headroom runs on the hour-of-day profile; LightGBM kept as a labelled offline ablation
- [x] `docs/FORECASTING.md`: full method for both
- [ ] Held-out benchmark + capacity study re-run on the profile (running)
- [ ] New numbers into README, `docs/VIDEO.md`, Proof tab; re-record the 5 operator answers (~$0.10)

## B. Headroom Check: any building, any utility (Impact, Demo, Agentic)
The answer to "does this work beyond your one site?"
- [ ] `headroom/check.py`: given any building's interval meter data (Green Button / CSV, hourly or
      15-min), a service size (or the code's 80% continuous-load rule), and a tariff, run the screening:
      hosting capacity with Headroom vs standard smart charging on real Palo Alto sessions, $/month,
      drivers served, violations
- [ ] Portfolio study `scripts/portfolio_study.py`: the screening on ~50 real BDG2 buildings across
      US sites (quality-filtered); headline "median +X% chargers on the same connection"; results CSV
- [ ] Agent tool `check_building` + app tab "Check my building": upload CSV, pick or upload tariff
      (Muse Spark reads an uploaded PDF; accepted only if it reproduces a bill total the owner types),
      answer card with chart
- [ ] Tests: screening invariants (0 violations, served share), CSV parser on hourly and 15-min inputs

## C. Meta models, end to end (Meta tooling, open-weight bonus)
- [ ] Voice: test the audio path on Muse Spark 1.3 with a generated test clip (< $0.01); owner's own
      voice clip for the video
- [ ] Glimmer failover recordings on Kaggle (owner runs the notebook, ~20 min; free)
- [ ] `scripts/package_kaggle.py`: builds the upload zip so the Kaggle step is one upload
- [ ] Provenance (staff asked): Proof tab lists every recorded call (model, time, tokens, cost, which
      verifier passed); recordings and traces in the repo

## D. App polish (Demo)
- [ ] First screen tells the story in 10 seconds: the $2,572 spike, the result, "try it" buttons
- [ ] Replace deprecated `st.components.v1.html`; no warnings in the console
- [ ] Every tab works in Replay with no key and no outside account; clear messages otherwise
- [ ] Fresh-clone test of the whole demo; full test suite green

## E. Repo ready to go public (pass/fail stage)
- [ ] README: built-during-hackathon vs. third-party (adacharge, acnportal) clearly separated; how to run
      the demo; provenance; models used; data and assumptions
- [ ] Remove the prior-work notebook from git history and the raw Palo Alto CSV (fetched by script):
      prepared by Claude, **force-push by the owner** (history rewrite)
- [ ] Hosted demo: Streamlit Community Cloud from the public repo (owner signs in once)

## F. Story (Demo, viral)
- [ ] `docs/VIDEO.md` updated: Headroom Check and the internet-cut failover become scenes
- [ ] Title cards rendered (PNG) from the final numbers
- [ ] Submission text and X post drafts (`docs/SUBMISSION.md`)

## Owner tasks (only you can do these)
1. Push after each block (`git push origin main`); the history-rewrite push in E is `git push --force`.
2. Kaggle: free account + phone verification, then upload the zip and run the notebook (C).
3. Record one spoken request on your phone (for the video).
4. Streamlit Community Cloud: sign in with GitHub and deploy `app.py` once the repo is public (E).
5. Optional but strong: 10–20 s of your own EV-bay footage (faces and plates blurred).
6. Record the video Nov 3 from `docs/VIDEO.md`; post on X by Nov 8.
