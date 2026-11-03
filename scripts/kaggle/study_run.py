"""Kaggle kernel: run a long study on Kaggle's CPUs and return its result files.

Pushed by scripts/kaggle_run.py (`study` mode); reads the project from the private dataset
<user>/headroom-code (a `git archive` of the repo: committed files only, never .env). No model and no
key is involved. Writes study_results.zip (new or changed files under data/results/) to the output.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

COMMAND = os.environ.get("STUDY_COMMAND", "scripts/portfolio_study.py run")
STUDY_ENV = os.environ.get("STUDY_ENV", "PORTFOLIO_SHARE=0.9")
WORK = "/kaggle/working/headroom"


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


src = glob.glob("/kaggle/input/**/kaggle_headroom.zip", recursive=True)
if src:
    with zipfile.ZipFile(src[0]) as z:
        z.extractall(WORK)
else:  # Kaggle auto-extracts uploaded zips
    root = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/app.py", recursive=True))
    shutil.copytree(root, WORK, dirs_exist_ok=True)
sh(f"cd {WORK} && pip install -q -r requirements.lock.txt")
sh("nproc; free -g")

results = f"{WORK}/data/results"
before = {p: os.path.getmtime(p) for p in glob.glob(f"{results}/**/*", recursive=True)}
env = dict(os.environ, PYTHONIOENCODING="utf-8", WORKERS=str(os.cpu_count() or 4),
           **dict(kv.split("=", 1) for kv in STUDY_ENV.split() if "=" in kv))
subprocess.run([sys.executable, *COMMAND.split()], cwd=WORK, env=env, check=False)

changed = [p for p in glob.glob(f"{results}/**/*", recursive=True)
           if os.path.isfile(p) and before.get(p) != os.path.getmtime(p)]
with zipfile.ZipFile("/kaggle/working/study_results.zip", "w") as z:
    for p in changed:
        z.write(p, os.path.relpath(p, WORK))
print(f"{len(changed)} result files -> study_results.zip", flush=True)
shutil.rmtree(WORK, ignore_errors=True)
