"""Blocked-bay detection: Muse Spark Open watches the bay video; charger data decides.

A meter can't see a bay that's occupied but not charging (a finished EV left parked, or a car that
never plugged in). The camera can. Spark Open gets the clip plus a still with the EV bays numbered,
and answers in strict JSON per bay. A bay is reported blocked only when the camera sees a vehicle
AND the charger for that bay reports no energy flowing; disagreements are flagged for a human.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from ..config import DATA
from ..meta.client import ModelClient, as_data_url

VIDEO = DATA / "video"


class Bay(BaseModel):
    number: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the still's pixels


class BayObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bay: int
    occupied: bool
    vehicle: str = Field(description="short description, e.g. 'white SUV', or 'none'")
    plugged_in: Literal["yes", "no", "unclear"] = Field(description="is a charging cable visibly connected")
    minutes_visible: float = Field(ge=0, description="how long a vehicle is in the bay during the clip")


class BayReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bays: list[BayObservation]


SYSTEM = """You watch a parking-lot camera for an EV-charging site. The still image marks EV charging bays
with numbered boxes. Watch the video and report every numbered bay. Only report what you can see;
use "unclear" when a cable isn't visible. Answer in the JSON schema exactly."""


def load_bays(name: str) -> list[Bay]:
    return [Bay(**b) for b in json.loads((VIDEO / f"{name}.bays.json").read_text(encoding="utf-8"))]


def annotate(still: Path, bays: list[Bay], out: Path) -> Path:
    image = Image.open(still).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for bay in bays:
        draw.rectangle(bay.box, outline=(0, 230, 118), width=4)
        x0, y0 = bay.box[0], bay.box[1]
        draw.rectangle((x0, y0 - 34, x0 + 34, y0), fill=(0, 230, 118))
        draw.text((x0 + 8, y0 - 33), str(bay.number), fill=(0, 0, 0), font=font)
    image.save(out)
    return out


def video_part(path: Path) -> dict:
    url = as_data_url(path)
    if os.environ.get("HEADROOM_VIDEO_PART", "video_url") == "input_video":
        return {"type": "input_video", "input_video": {"data": base64.b64encode(path.read_bytes()).decode(), "format": "mp4"}}
    return {"type": "video_url", "video_url": {"url": url}}


def observe(clip: Path, still: Path, bays: list[Bay], client: ModelClient | None = None) -> BayReport:
    client = client or ModelClient("spark")
    marked = annotate(still, bays, still.with_suffix(".bays.png"))
    response = client.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": [
             {"type": "text", "text": f"EV bays: {[b.number for b in bays]}. Still with bays marked:"},
             {"type": "image_url", "image_url": {"url": as_data_url(marked)}},
             {"type": "text", "text": "The video:"},
             video_part(clip)]}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "BayReport", "strict": True, "schema": BayReport.model_json_schema()}},
        reasoning_effort="low")
    return BayReport.model_validate_json(response["choices"][0]["message"]["content"])


def decide(report: BayReport, charger_kw: dict[int, float], min_minutes: float = 1.0) -> list[dict]:
    """Camera + charger -> decision per bay. charger_kw: bay -> power the charger reports now."""
    out = []
    for obs in report.bays:
        kw = charger_kw.get(obs.bay, 0.0)
        if obs.occupied and kw < 0.1 and obs.minutes_visible >= min_minutes:
            status = "blocked"  # a vehicle is there and no energy flows
            action = "remove the bay from the plan and notify the site manager"
        elif obs.occupied and kw >= 0.1:
            status = "charging"
            action = "none"
        elif not obs.occupied and kw >= 0.1:
            status = "check"  # charger says energy flows, camera sees no car: one of them is wrong
            action = "flag for a human: camera and charger disagree"
        else:
            status = "free"
            action = "none"
        out.append({"bay": obs.bay, "camera": obs.model_dump(), "charger_kw": kw, "status": status, "action": action})
    return out


def occupancy_accuracy(report: BayReport, name: str) -> dict:
    """Compare the model's occupancy calls with hand labels (data/video/<name>.gold.json)."""
    gold = json.loads((VIDEO / f"{name}.gold.json").read_text(encoding="utf-8"))["occupied"]
    seen = {str(o.bay): o.occupied for o in report.bays}
    hits = sum(seen.get(b) == v for b, v in gold.items())
    return {"bays": len(gold), "correct": hits, "accuracy": hits / len(gold),
            "missing": [b for b in gold if b not in seen]}
