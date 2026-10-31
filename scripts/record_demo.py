"""Run every model moment in the demo once, live, and save the responses for replay.

    python scripts/record_demo.py            # Muse Spark (HEADROOM_SPARK_* in .env)
    python scripts/record_demo.py glimmer    # Muse Glimmer   (HEADROOM_GLIMMER_* in .env)

Each step prints what the model did and whether the verifiers passed. Responses land in
data/recordings/; the control room replays them when "Model calls: Replay recorded" is selected.
The spend guard in headroom/meta/client.py applies.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headroom.agent.bays import decide, load_bays, observe, occupancy_accuracy  # noqa: E402
from headroom.agent.operator import DEMO_REQUESTS, SYSTEM, run  # noqa: E402
from headroom.agent.site import ReplaySite  # noqa: E402
from headroom.agent.tools import ReadTariff  # noqa: E402
from headroom.config import DATA  # noqa: E402
from headroom.meta.client import ModelClient, spent_usd  # noqa: E402

# the operator moments in docs/VIDEO.md, in order; RECORD_N=5 records only the first five
REQUESTS = DEMO_REQUESTS[: int(os.environ.get("RECORD_N", len(DEMO_REQUESTS)))]


def step(title: str, fn) -> None:
    print(f"\n=== {title}")
    try:
        fn()
    except Exception:  # noqa: BLE001 - keep recording the other moments
        traceback.print_exc()


def main(role: str) -> None:
    client = ModelClient(role, mode="auto")
    print(f"role={role} model={client.model} spent so far=${spent_usd():.2f}")
    site = ReplaySite(month="2017-06", n_ports=24)

    if role == "spark":
        for doc in ("cpau-e-4-tou-2026", "cpau-e-7-tou-2026"):
            def read(doc=doc):
                out = site.read_tariff(ReadTariff(document=doc))
                print(json.dumps(out.get("verification"), indent=1, default=str))
            step(f"tariff {doc}", read)

        def bays():
            report = observe(DATA / "video" / "lot_close_5fps.mp4", DATA / "video" / "lot_close.jpg",
                             load_bays("lot_close"), client=client)
            print(json.dumps(decide(report, {3: 6.1}), indent=1, default=str))
            print(occupancy_accuracy(report, "lot_close"))
        step("bays", bays)

    history = [{"role": "system", "content": SYSTEM}]
    for when, request in REQUESTS:
        def ask(when=when, request=request):
            nonlocal history
            site.now_local = when
            turn = run(site, request, history=history, client=client)
            history = turn.messages
            for s in turn.steps:
                print(("  tool " + s.tool + (" OK" if s.ok else " REJECTED")) if s.kind == "tool" else "  says: " + s.text[:200])
            print("  answer:", turn.answer[:600])
            print("  numbers not backed by a tool:", turn.unsupported or "none")
        step(f"operator: {request}", ask)
    print(f"\nspent so far: ${spent_usd():.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spark")
