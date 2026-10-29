# Fallback: Muse Glimmer on a free Kaggle GPU (no card)

Use only if no partner key arrives. Muse Glimmer is an open Meta model (Apache-2.0), so running it on
free hardware counts as using Meta's models. Muse Spark Open is too large for this.

## One-time setup (owner, ~10 minutes)
1. Create a free account at https://www.kaggle.com and **verify your phone number** (Settings → Phone verification). Without it, Kaggle gives no GPU.
2. Zip the project folder **without** `.venv`, `data/raw/bdg2/electricity_cleaned.csv` and `data/forecasts_v1` (keep `data/forecasts/`, `data/processed/`, `data/docs/`, `data/video/`).
3. Kaggle → **Datasets → New Dataset** → upload the zip → keep it **Private** → name it `headroom`.
4. Kaggle → **Code → New Notebook** → right panel: **Accelerator: GPU T4 x2**, **Internet: On** → **Add Input** → your `headroom` dataset.

## Notebook cells (paste one per cell, run in order)
```bash
# 1. Install Ollama and start it (uses both T4 GPUs)
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time; subprocess.Popen(["ollama", "serve"]); time.sleep(5)
!ollama pull muse-glimmer
```
```bash
# 2. Project
!cp -r /kaggle/input/headroom/* /kaggle/working/ && cd /kaggle/working && pip install -q -r requirements.lock.txt
```
```python
# 3. Point Headroom at the local Glimmer and record the demo
import os
os.environ.update({"HEADROOM_GLIMMER_BASE_URL": "http://localhost:11434/v1", "HEADROOM_GLIMMER_API_KEY": "ollama",
                   "HEADROOM_GLIMMER_MODEL": "muse-glimmer", "HEADROOM_BUDGET_USD": "1000000"})
!cd /kaggle/working && python scripts/record_demo.py glimmer
```
```bash
# 4. Package the recordings for download (Output panel → download)
!cd /kaggle/working && zip -r recordings.zip data/recordings
```
Then unzip `recordings.zip` into the project on the laptop; the control room replays them.
