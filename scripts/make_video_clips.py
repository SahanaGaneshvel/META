"""Cut the bay-camera clips from the public-domain source video (see data/video/SOURCES.md).

    python scripts/make_video_clips.py

Keeps only the segments where the camera is static (it pans and zooms between ~155 s and ~252 s),
re-encodes to H.264 MP4 with faststart at 5 fps (SAM 3.1 bills per frame; occupancy changes over
minutes), and saves a still from each segment for single-image tests.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "video"
SOURCE = VIDEO / "src" / "BLK-HDPTZ12_Security_Camera_Parking_Lot_Surveillance_Video.webm"

# name: (start s, end s, still at s) for the static camera segments
SEGMENTS = {"lot_wide": (1, 155, 30), "lot_close": (253, 316, 280)}


def ffmpeg(*args: str) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", *args], check=True)


def main():
    if not SOURCE.exists():
        sys.exit(f"missing {SOURCE}: run python scripts/fetch_data.py")
    for name, (a, b, still) in SEGMENTS.items():
        ffmpeg("-ss", str(a), "-to", str(b), "-i", str(SOURCE), "-an", "-vf", "fps=5,scale=1280:-2",
               "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(VIDEO / f"{name}_5fps.mp4"))
        ffmpeg("-ss", str(still), "-i", str(SOURCE), "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3",
               str(VIDEO / f"{name}.jpg"))
        print(f"{name}: {b - a} s -> {name}_5fps.mp4, {name}.jpg")


if __name__ == "__main__":
    main()
