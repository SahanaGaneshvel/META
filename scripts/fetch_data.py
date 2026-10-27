"""Download any raw dataset missing from data/raw/ (e.g. after a fresh clone).

    python scripts/fetch_data.py

The BDG2 electricity file (167 MB) is over GitHub's 100 MB limit, so it is never committed;
everything else is in the repo but is restored here too if deleted.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "HeadroomHackathon/0.1 (research prototype)"}
BDG2 = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"

FILES = {
    "data/raw/bdg2/electricity_cleaned.csv": f"{BDG2}/meters/cleaned/electricity_cleaned.csv",
    "data/raw/bdg2/metadata.csv": f"{BDG2}/metadata/metadata.csv",
    "data/raw/bdg2/weather.csv": f"{BDG2}/weather/weather.csv",
    "data/raw/paloalto_ev_sessions.csv":
        "https://data.paloalto.gov/datasets/194693-electric-vehicle-charging-station-usage-july-2011-dec-2020.download/",
    "data/video/src/BLK-HDPTZ12_Security_Camera_Parking_Lot_Surveillance_Video.webm":
        "https://upload.wikimedia.org/wikipedia/commons/1/14/BLK-HDPTZ12_Security_Camera_Parking_Lot_Surveillance_Video.webm",
}


def main():
    for rel, url in FILES.items():
        dst = ROOT / rel
        if dst.exists() and dst.stat().st_size > 0:
            print(f"have  {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch {rel} ...", flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(dst, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        print(f"      {dst.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
