# Cricket Win Probability Project — Development Log

**Project:** `cric-win-pred`  
**Goal:** Build an end-to-end T20 cricket win probability prediction system.

---

# Day 1 — Project Setup & Python Packaging

## Work Completed
- Set up the Python project structure.
- Created the `cricketwp` package using the `src` layout.
- Configured the development environment.
- Installed the project in editable mode.
- Added and executed a smoke test.

## Key Learnings
- Python package structure and imports.
- Editable installation using `pip install -e .`.
- Basic project testing with pytest.

## Status
Completed.

---

# Day 2 — Cricsheet Data Exploration

## Work Planned
- Download IPL and T20I match data from Cricsheet.
- Extract the JSON files.
- Inspect the JSON structure.
- Identify teams, venue, match outcome, deliveries, runs, extras, and wickets.

## Key Learnings
- Understanding structured cricket match data.
- Exploring nested JSON.
- Identifying the raw information required for win probability prediction.

## Status
Data exploration and ingestion milestones continued in subsequent days.

---

# Days 3–6 — Data Pipeline & State Construction

## Work Completed
- Built the Cricsheet JSON parsing pipeline.
- Parsed ball-by-ball match data.
- Processed IPL and T20I datasets.
- Generated `balls.parquet`.
- Generated `matches.parquet`.
- Constructed first-innings state datasets.
- Constructed second-innings chase state datasets.
- Added cumulative runs, wickets, and balls bowled.
- Created features such as:
  - `runs_needed`
  - `balls_left`
  - `wickets_in_hand`
  - `target`
  - `label`
- Corrected notebook working-directory and data-path issues.

## Data Artifacts
- `data/interim/balls.parquet`
- `data/interim/matches.parquet`
- `data/interim/first_states.parquet`
- `data/interim/chase_states.parquet`

## Key Learnings
- JSON parsing and data engineering.
- Parquet storage.
- Ball-by-ball state construction.
- Cumulative calculations and target labeling.
- Data leakage awareness.

## Status
Completed.

---

# Days 7–10 — Baselines, Evaluation & Model Development

## Work Completed
- Built Baseline 0 using training-set average win probability.
- Built Baseline 1 using Logistic Regression.
- Used four initial chase features:
  - `runs_needed`
  - `balls_left`
  - `wickets_in_hand`
  - `rrr`
- Implemented evaluation metrics:
  - Log Loss
  - Brier Score
  - AUC
  - Expected Calibration Error (ECE)
- Created the `evaluate()` function in `src/cricketwp/eval.py`.
- Added tests for evaluation metrics.
- Tracked model experiments using MLflow.
- Developed and tuned the XGBoost model.
- Evaluated the model using chronological validation data.
- Performed match trajectory sanity checks.

## Validation Results — Logistic Regression

Validation period: 2024

| Metric | Value |
|---|---:|
| Log Loss | 0.38018 |
| Brier Score | 0.12429 |
| AUC | 0.90957 |
| ECE | 0.04385 |

## Key Learnings
- Baseline models provide a benchmark for advanced models.
- Log Loss and Brier Score evaluate probability quality.
- AUC evaluates ranking ability.
- ECE evaluates calibration.
- Chronological validation helps prevent temporal leakage.
- MLflow tracks experiments and metrics.

## Status
Completed.

---

# Day 11 — Probability Calibration & Chase Prediction

## Work Completed

### 1. Isotonic Calibration
- Calibrated XGBoost win probabilities using Isotonic Regression.
- Compared raw and calibrated reliability curves.
- Evaluated calibration using Log Loss, Brier Score, AUC, and ECE.
- Exported isotonic calibration thresholds to JSON.
- Verified the exported calibration mapping using `np.interp`.

### Calibration Results

| Metric | Raw | Calibrated |
|---|---:|---:|
| Log Loss | 0.34673 | 0.34031 |
| Brier Score | 0.11073 | 0.10960 |
| AUC | 0.92409 | 0.92450 |
| ECE | 0.02256 | 0.000046 |

### 2. Terminal Rules
Implemented `apply_rules()` in:

`src/cricketwp/predict.py`

Rules:
- If `runs_needed <= 0`, batting team's win probability is 1.0.
- If no balls remain and runs are still needed, probability is 0.0.
- If no wickets remain and runs are still needed, probability is 0.0.

### 3. Chase Curve
Implemented `chase_curve()`.

Pipeline:
1. Generate raw XGBoost probabilities.
2. Apply isotonic calibration.
3. Apply terminal cricket rules.
4. Generate ball-by-ball win probability output.
5. Calculate WPA.

### 4. Win Probability Added (WPA)

WPA is the change in win probability from one ball to the next.

Formula:

WPA = Current Win Probability − Previous Win Probability

Implementation:

`out["wpa"] = out.p_chase.diff().fillna(0)`

Large positive or negative WPA values identify potentially important match moments.

### Testing
- `test_chase_completed`
- `test_no_balls_remaining`
- `test_all_out_before_chase_completed`
- `test_chase_curve_and_wpa`

Command:

`pytest tests/test_predict.py -v`

Result: **4 passed**

## Key Learnings
- Calibration makes probability estimates more trustworthy.
- Terminal rules should be handled outside the ML model.
- WPA measures the impact of a ball on win probability.
- Key moments can be identified using WPA.

## Status
Day 11 completed.


# Day 13 — GATE C Final Evaluation

**Date:** 2026-11-07  
**Project:** `cric-win-pred`  
**Focus:** Final model evaluation on held-out test data

## Today's Objectives

- Evaluate the final calibrated model on the held-out chase test set.
- Apply probability calibration using the validation-fitted isotonic calibration mapping.
- Apply deterministic chase outcome rules.
- Evaluate overall model performance.
- Evaluate performance across different chase phases.
- Record final metrics without tuning against the test set.

## Work Completed

- Generated raw chase-win probabilities using the trained classifier.
- Applied isotonic calibration using the saved calibration mapping.
- Applied deterministic prediction rules:
  - If runs needed are zero or less, chase-win probability is 1.0.
  - If balls remaining are zero while runs are still required, probability is 0.0.
  - If wickets in hand are zero while runs are still required, probability is 0.0.
- Evaluated the final probabilities using Log Loss, Brier Score, ROC-AUC, and ECE.
- Evaluated metrics separately across four chase phases.
- Recorded final test results in `results.md`.

## Final Overall Test Results

| Metric | Result |
|---|---:|
| Log Loss | 0.3217 |
| Brier Score | 0.1029 |
| ROC-AUC | 0.9377 |
| ECE | 0.0319 |

## Chase Phase Results

| Phase | Log Loss | Brier Score | ROC-AUC | ECE |
|---|---:|---:|---:|---:|
| (0, 36] | 0.4167 | 0.1362 | 0.8922 | 0.0418 |
| (36, 60] | 0.3387 | 0.1059 | 0.9337 | 0.0364 |
| (60, 90] | 0.2663 | 0.0842 | 0.9575 | 0.0298 |
| (90, 120] | 0.1845 | 0.0575 | 0.9792 | 0.0278 |

## Learning Reflection

The final model achieved strong test-set discrimination with a ROC-AUC of 0.9377.

The chase-phase evaluation showed that the model performed better in later phases than in the earliest phase. The early phase had higher logloss and Brier score, indicating that probability predictions were more challenging when more balls remained.

The overall ECE was 0.0319, so the model is not perfectly calibrated. Calibration quality should be interpreted alongside logloss, Brier score, and AUC.

The held-out test set was used for final evaluation rather than model selection. I will avoid tuning the model based on these test results.

## Day 13 Status

- [x] Final model test evaluation
- [x] Overall metrics recorded
- [x] Chase-phase metrics recorded
- [x] Results documented
- [x] Test set kept separate from model tuning

**Day 13 completed.**


## Day 14 — First-Innings Score Projection (Quantile Models)

**Date:** November 9, 2026

### Objective
Train three quantile regression models to predict a first-innings final-score range using the 10th, 50th, and 90th percentiles.

### Implementation
- Reused the existing first-innings train/validation split:
  - Training rows: 331,864
  - Validation rows: 75,896
- Used `FIRST_FEATURES` (12 features) and `final_total` as the target.
- Trained three XGBoost quantile models with quantiles 0.1, 0.5, and 0.9.
- Evaluated validation coverage and median prediction MAE.
- Evaluated MAE, coverage, and average prediction-band width by innings stage.
- Generated a predicted-median vs actual-final-total scatter plot for states at exactly 60 balls.
- Hand-tested five representative innings states.
- Saved the three trained models under `artifacts/`.

### Validation Results
- 10th–90th percentile coverage: **71.96%**
- Median prediction MAE: **19.50 runs**

### Stage-wise Results
| Balls Bowled | MAE (runs) | Coverage | Avg Band Width (runs) |
|---|---:|---:|---:|
| 0–36 | 30.02 | 0.683 | 76.21 |
| 36–60 | 22.14 | 0.688 | 58.46 |
| 60–90 | 15.60 | 0.725 | 42.16 |
| 90–120 | 7.54 | 0.790 | 22.63 |

### Sanity Checks
Five hand-tested states produced plausible median predictions and ordered score ranges:
- 0 balls at a high-scoring venue: median 165.0
- 60/1 after 10 overs: median 151.0
- 40/5 after 10 overs: median 97.7
- 150/3 after 17 overs: median 182.3
- 100/9 after 18 overs: median 107.5

The prediction band narrowed as the innings progressed, and median MAE decreased across stages.

### Artifacts
- `artifacts/score_q10.json`
- `artifacts/score_q50.json`
- `artifacts/score_q90.json`
- `docs/figs/first_innings_median_vs_actual_60_balls.png`

### Reflection
The median model became more accurate as more innings information became available. The prediction band narrowed through the innings, but overall coverage (71.96%) remained below the approximate 80% target, especially during early stages. This is a limitation to document and revisit during later model refinement.


## Day 15 — First-Innings Win Probability Model

### Work Completed
- Trained an XGBoost classifier for batting-first win probability.
- Applied monotonic constraints to runs and wickets in hand.
- Used early stopping to select the best iteration.
- Calibrated predicted probabilities using isotonic regression.
- Evaluated performance across innings phases on the test set.
- Compared the batting-first model with the chase model.
- Saved the model and calibration artifacts.

### Key Results
- Best iteration: 119
- Best validation logloss: 0.5683
- Calibrated validation logloss: 0.5601
- Test average cross-model consistency gap: 8.77 percentage points
- Matched test matches for consistency analysis: 1,133

### Learning Reflection
Isotonic regression improves the reliability of predicted probabilities by calibrating their relationship with observed outcomes.

Pairing predictions by match ID is essential when comparing two models. Directly comparing arrays without ensuring matching row order can produce misleading consistency results.

The consistency gap measures disagreement between models, not how often either model is correct.

### Next Steps
- Review model artifacts and documentation.
- Check that model loading works correctly.
- Continue with the next planned project session.
