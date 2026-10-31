# Forecasting: what Headroom runs on, and what we compared it with

Headroom's planner needs, every 15 minutes, a forecast of the building's load for the next 24 hours
as quantiles (P10, P50, P90, P99). It only charges cars in the room the P99 leaves under the demand
level the month has already paid for.

**At runtime Headroom uses a statistical profile, not a trained model.** The only models called at
runtime are Meta's (Muse Spark, Muse Glimmer). A LightGBM forecaster was built and evaluated as an
offline ablation; its full method is below so the comparison can be reproduced.

## 1. Runtime: hour-of-day profile (`headroom/forecast/profile.py`)
Issued at the start of interval t, using only meter readings up to t-1:

1. **Profile.** For the target's local 15-minute clock slot and day type (workday, or weekend/US
   holiday), the mean load over the last 28 days of that type before the issue day.
2. **Persistence correction.** Add the latest error (measured minus profile at t-1), decayed by
   phi^(h+1) for horizon h. phi is the lag-1 autocorrelation of the errors on the fitting period,
   clipped to [0, 0.999].
3. **Quantiles.** Add the empirical quantile of the forecast's own errors for that horizon, measured
   out of sample on the fitting period (each day's profile only uses earlier days).

Walk-forward: forecasts for month M are fitted on everything before M (`scripts/precompute_forecasts.py`)
and saved to `data/forecasts/M.npz` with the month's coverage in `M.json`. Seconds per month on a
laptop, no GPU.

**Calibration on the held-out test months (May–Dec 2017):** P90 covered 86–95% (target 90%), P99
97.7–100% (target 99%). Files: `data/forecasts/2017-*.json`.

## 2. Ablation: LightGBM blend (`headroom/forecast/quantile_gbm.py`, `--model lgbm`)
Built to test whether a trained model would let Headroom charge more. It doesn't, materially (see the
README results), so it is not used at runtime.

- **Target:** the residual of the load from the profile above (the trees learn what the profile misses).
- **Rows:** one per (issue time t, horizon h), horizon as a feature (direct multi-horizon), 0–95 steps.
- **Features (all measured before t; no future weather):** h, target clock slot, day of week, workday
  flag, month, profile value at the target, last reading, last profile error, mean and slope of the
  last 4 readings, the same slot one day and one week earlier (only if already measured), last
  temperature, and the maximum temperature over the past 24 h.
- **Model:** one LightGBM quantile regressor per level (0.1, 0.5, 0.9, 0.99): 300 trees, learning
  rate 0.05, 63 leaves, min 200 samples per leaf, 80% row subsampling, 90% column subsampling.
- **Training data:** the site's own metered building load (UC Berkeley, BDG2) from Jan 2016 up to two
  months before the forecast month; issue times every 8 steps.
- **Calibration:** one-sided split conformal per quantile and horizon bucket (<1 h, 1–4 h, 4–12 h,
  12–24 h) on the two months before the forecast month, then re-calibrated every day on the previous
  28 days of its own errors (`RollingConformal`, past data only).
- **Blend:** the profile for the first 4 hours, LightGBM beyond, with the P99 taken as the larger of
  the two. The 4-hour switch was chosen on validation months (Mar–Apr 2017, pinball loss).
- **Splits:** train 2016, validation Jan–Apr 2017, test May–Dec 2017; nothing was tuned on test.
- **Reproduce:** `python scripts/precompute_forecasts.py 2017-05 2017-06 ... --model lgbm` writes
  `data/forecasts_lgbm/`; `scripts/final_benchmark.py` runs it as `headroom_lightgbm`.

## 3. Result of the comparison
On the held-out benchmark (May–Oct 2017, 24 chargers, 18 month-scenarios) the two give practically the
same result: $2,888.91 (profile) vs $2,889.62 (LightGBM) per month in added demand charges, 100% of
energy delivered, 0 violations (`data/results/final_summary.csv`, rows `headroom` and `headroom_lightgbm`).

Why: the forecasts are identical for the first 4 hours (the blend uses the profile there, chosen on
validation) and differ beyond (mean absolute difference 13–16 kW at 4–24 h in June 2017). The planner
re-plans every 15 minutes and what it commits is driven by the next few hours, so the plans coincide.
The value is in seeing the building at all, not in the forecasting model's complexity, so Headroom
ships the profile: simpler, faster, and no trained model at runtime.

Earlier full results with LightGBM as the main forecaster are kept in `data/results/*_lightgbm_v1.csv`.
