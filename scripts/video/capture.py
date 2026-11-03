"""Screenshots of the real app for the demo video, taken by driving it in a browser.

    python scripts/video/capture.py         # -> video/shots/*.png (1920x1080)

Starts the app on a local port in Replay mode, then walks the demo path: the heat-wave day hour by
hour, the recorded operator questions, the internet cut, Headroom Check, the tariff check and the
OCPP proof. Needs Playwright (scripts/video/requirements-video.txt) and its Chromium.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "video" / "shots"
PORT = 8599
URL = f"http://localhost:{PORT}"


def idle(page: Page, extra: float = 0.8) -> None:
    """Wait until Streamlit has finished re-running the script."""
    try:
        page.wait_for_load_state("networkidle", timeout=5_000)
    except Exception:  # noqa: BLE001 - the app keeps a websocket open; the status widget is the real signal
        pass
    time.sleep(0.5)
    # done = no spinner, no loading placeholder, no element faded out as stale by a rerun
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stSpinner\"], "
                           "[data-testid=\"stSkeleton\"], [data-stale=\"true\"]') && "
                           "[...document.images].every(i => i.complete)", timeout=180_000)
    time.sleep(extra)


def shot(page: Page, name: str) -> None:
    page.mouse.move(1910, 1070)  # keep the cursor out of the frame
    page.screenshot(path=str(SHOTS / f"{name}.png"))
    print("shot", name, flush=True)


def scroll_to(page: Page, selector: str, index: int = -1, block: str = "start") -> None:
    page.evaluate("([s, i, b]) => { const e = document.querySelectorAll(s); const el = e[i < 0 ? e.length + i : i];"
                  " if (el) el.scrollIntoView({block: b}); }", [selector, index, block])
    time.sleep(0.6)


def ask_all(page: Page, prefix: str, keep: dict[int, str]) -> None:
    """Click through the recorded questions; screenshot the answers listed in `keep` (1-based)."""
    messages = '[data-testid="stChatMessage"]'
    for i in range(1, 7):
        before = page.locator(messages).count()
        page.locator("button", has_text=re.compile(r"Ask:")).first.click()
        page.wait_for_function(f"() => document.querySelectorAll('{messages}').length >= {before + 2}",
                               timeout=180_000)
        idle(page, 1.2)
        if i in keep:
            scroll_to(page, '[data-testid="stChatMessage"]', -2)
            shot(page, f"{prefix}_{keep[i]}")


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HEADROOM_MODEL_MODE="replay", PYTHONIOENCODING="utf-8")
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            new = lambda: browser.new_page(viewport={"width": 1920, "height": 1080})  # noqa: E731
            page = new()
            for _ in range(60):
                try:
                    page.goto(URL)
                    break
                except Exception:  # noqa: BLE001 - the server is still starting
                    time.sleep(1)
            idle(page, 2)
            shot(page, "overview_top")
            scroll_to(page, "h5", 1)
            shot(page, "overview_steps")

            # the heat-wave day, hour by hour (the page opens at ?at=HH:MM)
            for minutes in range(14 * 60, 21 * 60 + 1, 15):
                page.goto(f"{URL}/replay?at={minutes // 60:02d}:{minutes % 60:02d}")
                idle(page, 0.5)
                page.wait_for_selector(".js-plotly-plot .scatterlayer .trace", timeout=120_000)  # chart drawn
                time.sleep(1.2)
                shot(page, f"replay_{minutes // 60:02d}{minutes % 60:02d}")

            # the recorded conversation with Muse Spark
            page = new()
            page.goto(f"{URL}/operator")
            idle(page, 1.5)
            ask_all(page, "spark", {1: "status", 3: "vans", 5: "expansion", 6: "buildings"})
            page.get_by_text(re.compile(r"How it got this")).last.click()
            time.sleep(1)
            scroll_to(page, '[data-testid="stChatMessage"]', -1, "end")
            shot(page, "spark_tools")

            # the internet cut: Muse Glimmer answers
            page = new()
            page.goto(f"{URL}/operator")
            idle(page, 1.5)
            page.get_by_text("Cut the site's internet").click()
            idle(page, 1.5)
            page.locator("button", has_text=re.compile(r"Ask:")).first.click()
            page.wait_for_function("() => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= 2",
                                   timeout=180_000)
            idle(page, 1.5)
            shot(page, "glimmer_status")

            # Headroom Check
            page = new()
            page.goto(f"{URL}/check_building")
            idle(page, 1.5)
            shot(page, "check_top")
            page.get_by_text("See every building").click()
            time.sleep(1.5)
            scroll_to(page, '[data-testid="stExpander"]', 0)
            shot(page, "check_buildings")
            page.get_by_role("button", name=re.compile("Try a sample building")).click()
            idle(page, 1.5)
            scroll_to(page, '[data-testid="stMetric"]', -1, "center")
            shot(page, "check_sample")

            # the tariff check on PG&E's tariff, and a misread being rejected
            page = new()
            page.goto(f"{URL}/verified")
            idle(page, 1.5)
            page.get_by_text("PG&E B-19", exact=True).click()
            idle(page, 1)
            page.get_by_role("button", name="Read and verify").click()
            page.get_by_text("Fields read correctly").wait_for(timeout=180_000)
            idle(page, 1.5)
            shot(page, "tariff_pge")
            page.get_by_text("What happens when the model misreads one number").click()
            time.sleep(1)
            shot(page, "tariff_misread")

            # proof: OCPP
            page = new()
            page.goto(f"{URL}/proof")
            idle(page, 1.5)
            page.get_by_role("tab", name="Charger control (OCPP)").click()
            page.get_by_text("Commands accepted").wait_for(timeout=60_000)
            page.get_by_text("One command, as sent").click()
            idle(page, 2)
            shot(page, "proof_ocpp")
            page.get_by_role("tab", name="Benchmark").click()
            idle(page, 2)
            shot(page, "proof_benchmark")
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
