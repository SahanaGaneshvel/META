"""Parse SAM 3.1 streamed output into per-frame bounding boxes.

Format (dev.meta.ai/docs/sam/reading-segmentation): each line is `<Nf>` (frame index) followed by
records `ID <|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..[;c=..]|><|mask;...|>`. Mask payloads are base85
and may contain commas or `|`, so only the purely numeric box tokens are matched; IDs are read as the
integer immediately before each box.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FRAME = re.compile(r"^<(\d+)f>")
_BOX = re.compile(r"(\d+)\s*,?\s*<\|box;([a-z0-9=;.\-]+)\|>")


@dataclass(frozen=True)
class Box:
    object_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    frame_w: float
    frame_h: float
    confidence: float | None = None

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


def parse_boxes(text: str) -> dict[int, list[Box]]:
    """Return {frame_index: [Box, ...]}; frames without detections are absent."""
    frames: dict[int, list[Box]] = {}
    for line in text.splitlines():
        line = line.strip()
        head = _FRAME.match(line)
        if not head:
            continue
        boxes = []
        for object_id, fields in _BOX.findall(line[head.end():]):
            kv = {k: float(v) for k, v in (part.split("=", 1) for part in fields.split(";") if "=" in part)}
            boxes.append(Box(int(object_id), kv["x1"], kv["y1"], kv["x2"], kv["y2"], kv["w"], kv["h"], kv.get("c")))
        frames[int(head.group(1))] = boxes
    return frames
