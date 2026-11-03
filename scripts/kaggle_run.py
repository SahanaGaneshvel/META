"""Run Headroom jobs on Kaggle's free machines, end to end from this machine.

    python scripts/kaggle_run.py            # Muse Glimmer on a T4 x2: record the operator's answers
    RECORD_N=6 python scripts/kaggle_run.py # include the last question once the portfolio study is in
    python scripts/kaggle_run.py study      # a long study on Kaggle's CPUs (default: the connection-size
                                            # sensitivity, PORTFOLIO_SHARE=0.9 portfolio_study.py run)

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
DATASET = "headroom-code"


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


JOBS = {  # kind: (kernel, script, accelerator, output archive, files to unpack)
    "glimmer": ("headroom-glimmer", "glimmer_record.py", "NvidiaTeslaT4", "glimmer_recordings.zip", "data/recordings/"),
    "study": ("headroom-study", "study_run.py", None, "study_results.zip", "data/results/"),
}


def main(kind: str = "glimmer") -> None:
    kernel, script, accelerator, archive_name, prefix = JOBS[kind]
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

    # 2. the kernel (GPU for Glimmer, CPU for studies), with internet for pip
    kdir = work / "kernel"
    kdir.mkdir()
    code = (ROOT / "scripts" / "kaggle" / script).read_text()
    code = code.replace('os.environ.get("RECORD_N", "5")', f'"{os.environ.get("RECORD_N", "5")}"')
    for var, default in (("STUDY_COMMAND", "scripts/portfolio_study.py run"), ("STUDY_ENV", "PORTFOLIO_SHARE=0.9")):
        code = code.replace(f'os.environ.get("{var}", "{default}")', f'"{os.environ.get(var, default)}"')
    (kdir / script).write_text(code)
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{kernel}", "title": kernel, "code_file": script, "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": accelerator is not None, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET}"], "competition_sources": [], "kernel_sources": []}))
    print(kaggle("kernels", "push", "-p", str(kdir), *(["--accelerator", accelerator] if accelerator else [])))

    # 3. wait
    started = time.time()
    while True:
        status = kaggle("kernels", "status", f"{user}/{kernel}", check=False)
        print(f"[{(time.time() - started) / 60:4.1f} min] {status.strip()}", flush=True)
        if any(s in status.lower() for s in ("complete", "error", "cancel")):
            break
        time.sleep(60)

    # 4. download and unpack the results
    out = work / "output"
    print(kaggle("kernels", "output", f"{user}/{kernel}", "-p", str(out), check=False))
    log = next(out.glob("*.log"), None)
    if log:
        print(log.read_text(errors="replace")[-3000:])
    archive = out / archive_name
    if not archive.exists():
        raise SystemExit(f"nothing came back; kernel output is in {out}")
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.startswith(prefix) and ".." not in n]
        z.extractall(ROOT, members=names)
    print(f"{len(names)} files added under {prefix}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "glimmer")
