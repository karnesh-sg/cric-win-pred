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
