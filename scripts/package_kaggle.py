"""Build the zip to upload to Kaggle for the Muse Glimmer recordings (docs/GLIMMER_KAGGLE.md).

    python scripts/package_kaggle.py        # -> kaggle_headroom.zip (about 30 MB)

Made with `git archive` from the last commit, so it holds committed files only: never .env or a key.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_headroom.zip"

subprocess.run(["git", "archive", "--format=zip", "-o", str(OUT), "HEAD"], cwd=ROOT, check=True)
print(f"{OUT.name}: {OUT.stat().st_size / 1e6:.0f} MB (committed files only; no .env)")
