# Muse Glimmer on a free Kaggle GPU: record the "internet down" answers

Muse Glimmer is Meta's open-weight model (Apache-2.0, 30B). It plays the site's own on-premise operator:
when the internet link is cut, the app fails over to it. It needs ~18 GB of GPU memory, so we record its
answers once on Kaggle's free T4 ×2 and the app replays them. No API key is involved and nothing is billed.

## On the laptop (1 minute)
```powershell
.\.venv\Scripts\python.exe scripts\package_kaggle.py
```
This makes `kaggle_headroom.zip` from the last commit only (never `.env`).

## On Kaggle (about 20 minutes, mostly the model download)
1. Free account at https://www.kaggle.com, then **Settings → Phone verification** (needed for GPUs).
2. **Datasets → New Dataset** → upload `kaggle_headroom.zip` → **Private** → name it `headroom`.
3. **Code → New Notebook** → right panel: **Accelerator: GPU T4 x2**, **Internet: On** → **Add Input** → `headroom`.
4. Paste each block into its own cell and run them in order:

```python
# 1. Ollama with Muse Glimmer
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"]); time.sleep(8)
!ollama pull muse-glimmer
```
```python
# 2. The project
!mkdir -p /kaggle/working/headroom && cd /kaggle/working/headroom && unzip -q -o /kaggle/input/headroom/kaggle_headroom.zip || cp -r /kaggle/input/headroom/* /kaggle/working/headroom/
!cd /kaggle/working/headroom && pip install -q -r requirements.lock.txt
```
```python
# 3. Record the operator's answers with Glimmer (local model: free, no key)
import os
os.environ.update({"HEADROOM_GLIMMER_BASE_URL": "http://localhost:11434/v1", "HEADROOM_GLIMMER_API_KEY": "ollama",
                   "HEADROOM_GLIMMER_MODEL": "muse-glimmer", "HEADROOM_MODEL_MODE": "auto"})
!cd /kaggle/working/headroom && python scripts/record_demo.py glimmer
```
```python
# 4. Package the new recordings
!cd /kaggle/working/headroom && zip -q -r /kaggle/working/glimmer_recordings.zip data/recordings
```
5. **Output** panel (right) → download `glimmer_recordings.zip`.

## Back on the laptop
Unzip it into the project folder (it adds files to `data/recordings/`), then tell Claude: it checks the
answers with the number check and commits them. In the app, **Cut the site's internet** now replays
Glimmer's real answers.
