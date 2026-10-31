"""Kaggle kernel: run Muse Glimmer (open weights) under Ollama on T4 x2 and record the operator's answers.

Pushed by scripts/kaggle_run.py; reads the project from the private dataset <user>/headroom-code (a
`git archive` of the repo: committed files only, never .env). Writes glimmer_recordings.zip to the
kernel output. No API key is involved: the model runs on Kaggle's GPU.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

N_REQUESTS = os.environ.get("RECORD_N", "5")
WORK = "/kaggle/working/headroom"


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. the project
src = glob.glob("/kaggle/input/**/kaggle_headroom.zip", recursive=True)
if src:
    with zipfile.ZipFile(src[0]) as z:
        z.extractall(WORK)
else:  # Kaggle auto-extracts uploaded zips
    root = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/app.py", recursive=True))
    shutil.copytree(root, WORK, dirs_exist_ok=True)
before = set(os.listdir(f"{WORK}/data/recordings"))
sh(f"cd {WORK} && pip install -q -r requirements.lock.txt || pip install -q pandas numpy scipy pydantic openai "
   "python-dotenv holidays pypdf cvxpy clarabel acnportal 'adacharge @ git+https://github.com/caltech-netlab/adacharge.git' "
   "pyarrow requests highspy")

# 2. Ollama + Muse Glimmer on both T4s
sh("curl -fsSL https://ollama.com/install.sh | sh")
server = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
time.sleep(10)
sh("ollama pull muse-glimmer")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

# 3. record (local model: free)
env = dict(os.environ, HEADROOM_GLIMMER_BASE_URL="http://localhost:11434/v1", HEADROOM_GLIMMER_API_KEY="ollama",
           HEADROOM_GLIMMER_MODEL="muse-glimmer", HEADROOM_MODEL_MODE="auto", RECORD_N=N_REQUESTS,
           PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, "scripts/record_demo.py", "glimmer"], cwd=WORK, env=env, check=False)

# 4. package only the new recordings
new = sorted(set(os.listdir(f"{WORK}/data/recordings")) - before)
with zipfile.ZipFile("/kaggle/working/glimmer_recordings.zip", "w") as z:
    for name in new:
        z.write(f"{WORK}/data/recordings/{name}", f"data/recordings/{name}")
print(f"{len(new)} new recordings -> glimmer_recordings.zip", flush=True)
shutil.rmtree(WORK, ignore_errors=True)  # keep the kernel output small
server.terminate()
