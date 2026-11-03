# Muse Glimmer on a free Kaggle GPU: the "internet down" answers

Muse Glimmer is Meta's open-weight model (Apache-2.0, 30B). It plays the site's own on-premise operator:
when the internet link is cut, the app fails over to it. It needs about 18 GB of GPU memory, so its answers
are recorded on Kaggle's free T4 ×2 and the app replays them. No API key is involved and nothing is billed.

## One command
```powershell
$env:RECORD_N="6"; .\.venv\Scripts\python.exe scripts\kaggle_run.py
```
It needs a Kaggle API token in `~/.kaggle/` (a phone-verified account for GPUs). The script then:
1. Packages the last commit (`git archive`: committed files only, never `.env`) as a private Kaggle dataset.
2. Runs `scripts/kaggle/glimmer_record.py` on a T4 ×2. That script installs Ollama, pulls `muse-glimmer` and runs `scripts/record_demo.py glimmer`.
3. Waits for the run, then unpacks the new recordings into `data/recordings/`.

In the app, **Cut the site's internet** then replays Glimmer's real answers. Each answer goes through the
same tools and the same number check as Muse Spark's.

The same runner takes long studies to Kaggle's CPUs: `python scripts/kaggle_run.py study`.
