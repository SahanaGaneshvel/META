# Setting up Headroom on a new laptop

About 15 minutes, most of it downloads. Tested on Windows 11.

## 1. Install prerequisites
- **Python 3.12** (python.org; tick "Add to PATH"). Other versions may not match the pinned packages.
- **Git** (git-scm.com). Needed to clone the repo, and pip uses it to install `adacharge`.
- About **3 GB** free disk (environment ~2 GB, data ~400 MB).
- Read `docs/PROJECT_CONTEXT.md` first: it has every decision, finding and gotcha so far.

## 2. Get the code (the repo is private, so the owner must add you as a collaborator first)
```powershell
git clone https://github.com/SahanaGaneshvel/META-VS.git
cd META-VS
```

## 3. Create the environment
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```
If `torch` fails, install it first with
`.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, then rerun the line above.

## 4. Restore the one file that's too big for GitHub
```powershell
.\.venv\Scripts\python.exe scripts\fetch_data.py
```
It downloads the 167 MB BDG2 electricity file; everything else is already in the repo.

## 5. Secrets
```powershell
copy .env.example .env
```
Put the Meta API key in `.env` as `MODEL_API_KEY=...`. **Never commit `.env` or paste the key into chat.**

## 6. Verify everything works
```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe -m headroom.data.pipeline       # rebuilds data/processed (~5 s)
.\.venv\Scripts\python.exe -m pytest -q                     # 12 tests pass
.\.venv\Scripts\python.exe scripts\baselines_check.py        # least_laxity and adacharge_mpc show 0 violations
.\.venv\Scripts\python.exe scripts\api_smoke.py              # needs MODEL_API_KEY; all checks PASS
```
