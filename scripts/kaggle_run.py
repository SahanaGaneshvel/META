"""Record Muse Glimmer's answers on a free Kaggle T4 x2, end to end from this machine.

    python scripts/kaggle_run.py            # package, upload (private), run, wait, download
    RECORD_N=6 python scripts/kaggle_run.py # include the last question once the portfolio study is in

Needs a Kaggle API token in ~/.kaggle/access_token (phone-verified account for GPUs). The upload is a
`git archive` of the last commit: committed files only, never .env. Nothing here is billed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(Path(sys.executable).with_name("kaggle"))
DATASET, KERNEL = "headroom-code", "headroom-glimmer"


def kaggle(*args: str, check: bool = True) -> str:
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                         cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if check and out.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout + out.stderr


def username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username") or api.read_config_file().get("username")


def main() -> None:
    user = username()
    print("Kaggle user:", user)
    work = Path(tempfile.mkdtemp())

    # 1. the project, as a private dataset
    data_dir = work / "dataset"
    data_dir.mkdir()
    subprocess.run(["git", "archive", "--format=zip", "-o", str(data_dir / "kaggle_headroom.zip"), "HEAD"],
                   cwd=ROOT, check=True)
    (data_dir / "dataset-metadata.json").write_text(json.dumps(
        {"title": "headroom-code", "id": f"{user}/{DATASET}", "licenses": [{"name": "other"}]}))
    created = kaggle("datasets", "create", "-p", str(data_dir), check=False)
    print(created)
    if "already" in created.lower() or "error" in created.lower():
        print(kaggle("datasets", "version", "-p", str(data_dir), "-m", "update"))
    time.sleep(30)  # let the dataset finish processing

    # 2. the kernel, on T4 x2 with internet
    kdir = work / "kernel"
    kdir.mkdir()
    code = (ROOT / "scripts" / "kaggle" / "glimmer_record.py").read_text()
    code = code.replace('os.environ.get("RECORD_N", "5")', f'"{os.environ.get("RECORD_N", "5")}"')
    (kdir / "glimmer_record.py").write_text(code)
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL}", "title": KERNEL, "code_file": "glimmer_record.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET}"], "competition_sources": [], "kernel_sources": []}))
    print(kaggle("kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4"))

    # 3. wait
    started = time.time()
    while True:
        status = kaggle("kernels", "status", f"{user}/{KERNEL}", check=False)
        print(f"[{(time.time() - started) / 60:4.1f} min] {status.strip()}", flush=True)
        if any(s in status.lower() for s in ("complete", "error", "cancel")):
            break
        time.sleep(60)

    # 4. download and unpack the new recordings
    out = work / "output"
    print(kaggle("kernels", "output", f"{user}/{KERNEL}", "-p", str(out), check=False))
    log = next(out.glob("*.log"), None)
    if log:
        print(log.read_text(errors="replace")[-3000:])
    archive = out / "glimmer_recordings.zip"
    if not archive.exists():
        raise SystemExit(f"no recordings came back; kernel output is in {out}")
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.startswith("data/recordings/") and n.endswith(".json")]
        z.extractall(ROOT, members=names)
    print(f"{len(names)} Glimmer recordings added to data/recordings/")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
