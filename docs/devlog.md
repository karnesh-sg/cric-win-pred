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


# Day 16 — Development Log

- Created replay artifact from test-period matches.
- Loaded replay data into the FastAPI application.
- Added match listing and match replay endpoints.
- Verified replay output using API requests.
- Created API tests using FastAPI `TestClient`.
- Fixed response handling by parsing responses with `.json()`.
- Ran the full test suite: **17 passed**.

## Learning
- API testing with `TestClient`
- HTTP 422 validation
- Probability invariants
- Monotonicity checks
- JSON response parsing

## Reflection
All API tests passed. A Starlette deprecation warning remains but does not affect test execution.

# Day 17 — Results

## Completed
- Built Streamlit live scenario dashboard.
- Integrated both innings with FastAPI.
- Added win probability bar and prediction factors.
- Added first-innings projected score display.
- Added overs-to-balls conversion.
- Added API error handling.

## Status
Local API integration working.

## Day 18 — Ball-by-Ball Match Replay

### Objective
Build an interactive match replay system to visualize how win probability changes throughout a T20 match.

### Implementation
- Integrated ball-by-ball replay data into the Streamlit dashboard.
- Added match selection functionality.
- Implemented a replay slider to navigate through deliveries.
- Added a Play button for match playback.
- Visualized batting-first team win probability throughout the match.
- Displayed the opposing team's probability as `1 - p`.
- Added event markers for:
  - Wickets
  - Sixes
  - Important match moments
- Added a vertical marker representing the current delivery.

### Issues resolved
- Fixed missing `over` column in key-moment calculations.
- Fixed undefined `teams` variable in replay mode.
- Improved replay state handling and chart updates.

### Verification
- Replay slider verified.
- Play functionality verified.
- Probability graph and event markers visually checked.

**Status: Completed**

---

## Day 19 — Dockerization and CI Integration

### Objective
Containerize the prediction API and establish automated testing through CI.

### Implementation
- Created Docker configuration for the FastAPI application.
- Built the Docker image.
- Started the API container.
- Resolved host-port conflicts between existing containers.
- Ran the `cric-win-pred-api` container on host port `8001`, while the existing `cric-api` used port `8000`.
- Verified API health.
- Verified model metadata through `/model-info`.
- Tested prediction endpoints.
- Checked Swagger documentation.
- Reviewed container logs.
- Verified GitHub Actions workflow.

### Verification
- Docker container started successfully.
- `/health` returned a healthy status.
- Model version `1.0.0` confirmed.
- Prediction endpoints responded successfully.
- Swagger UI accessible.
- GitHub Actions passed.

**Status: Completed**

---

## Day 20 — API Deployment and Integration

### Objective
Make the trained prediction system accessible through a deployed API.

### Implementation
- Continued deployment and integration work for the FastAPI prediction service.
- Verified the deployed API.
- Checked chase prediction endpoint.
- Checked first-innings prediction endpoint.
- Verified API response structure and model loading.
- Investigated deployment/runtime issues and corrected integration problems.

### Deployment

Production API:

https://cric-win-pred.onrender.com

### Verification
- Deployed API responded successfully.
- Chase prediction endpoint tested.
- First-innings prediction endpoint tested.
- Model version and prediction responses checked.

**Status: Completed**

---

## Day 21 — First-Innings Prediction and Dashboard Integration

### Objective
Integrate first-innings score prediction into the cricket analytics application.

### Implementation
- Integrated the trained first-innings quantile prediction models.
- Added Q10, Q50 and Q90 score projections to the dashboard.
- Connected the Streamlit frontend with the FastAPI backend.
- Improved the presentation of predicted score ranges.
- Ensured the dashboard uses actual trained quantile-model predictions rather than a manually calculated heuristic.

### Quantile interpretation

- **Q10:** Lower estimated score boundary.
- **Q50:** Median predicted score.
- **Q90:** Upper estimated score boundary.

These represent conditional score estimates, not guaranteed outcomes.

### Verification
- First-innings prediction endpoint checked.
- Q10, Q50 and Q90 values verified in the dashboard.
- Dashboard/API integration checked.

**Status: Completed**

---

## Day 22 — Tarazu Dashboard and UI/UX Improvements

### Objective
Improve the dashboard into a professional cricket intelligence platform.

### Implementation
- Developed the premium dark dashboard interface.
- Established the branding:

  **Tarazu · तराज़ू · தராசு**

  **CRICKET INTELLIGENCE PLATFORM**

- Improved dashboard layout and visual hierarchy.
- Integrated live match-state prediction.
- Added probability visualization.
- Integrated first-innings score projections.
- Integrated match replay.
- Added model explanation visualizations.
- Improved scenario analysis and dashboard interactions.
- Fixed UI rendering and state-management issues.

### Verification
- Dashboard loaded successfully.
- Live scenario interaction checked.
- Dynamic UI updates verified.
- First-innings and replay sections checked.

**Status: Completed**

---

## Day 23 — What-If Simulator and Model Trust Panel

### Objective
Build an interactive next-ball simulator and improve model transparency.

### 1. What-If Prediction Engine

Implemented a hypothetical next-delivery prediction system for the second innings.

Six outcomes are evaluated:

1. Dot ball
2. One run
3. Two runs
4. Four runs
5. Six runs
6. Wicket

For each hypothetical outcome, the system calculates:

- Batting team's current win probability.
- Win probability after the hypothetical delivery.
- Change in win probability.

### 2. FastAPI Integration

Implemented endpoint:

`POST /what-if/chase`

The endpoint:
- Accepts the current match state.
- Evaluates six hypothetical legal-ball outcomes.
- Returns the resulting win probabilities.
- Calculates the difference from the current probability.
- Handles terminal match states.

### 3. Streamlit Integration

- Integrated all six outcomes into the Tarazu dashboard.
- Added individual outcome cards.
- Displayed probability changes in percentage points.
- Ensured the what-if results update when the user changes the match state and analyses the scenario.
- Preserved the base prediction if the what-if request fails.

### 4. Model Trust and Transparency Panel

Added a dedicated Model Trust Panel.

The panel displays the recorded held-out chase-model test metrics:

| Metric | Value |
|---|---:|
| ROC AUC | 0.9377 |
| Log Loss | 0.3217 |
| Brier Score | 0.1029 |
| ECE | 0.0319 |

Model metadata:
- Version: `1.0.0`
- Metadata date: `2026-11-09`
- Calibration: Isotonic calibration.

The panel also explains:

- Meaning of evaluation metrics.
- Probability calibration.
- First-innings quantile predictions.
- TreeSHAP feature contributions.
- What-if simulator assumptions.
- Model limitations and intended use.

### 5. Technical Accuracy and Transparency

Documented important distinctions:

- Chase test metrics must not be presented as first-innings test metrics.
- TreeSHAP values explain raw model margin/log-odds, not direct calibrated probability percentage-point changes.
- What-if outcomes are conditional hypothetical estimates, not probabilities of those delivery outcomes occurring.
- The what-if simulation considers hypothetical legal deliveries.

### 6. Final Testing

Executed:

`pytest -q`

Result:

`21 passed, 1 warning in 7.42s`

The warning concerns Starlette's deprecated use of `httpx` with `starlette.testclient`, recommending `httpx2`.

No test failures.

### Manual Verification
- What-if outcome cards verified.
- Dynamic updates verified.
- Model Trust Panel visually verified.
- First-innings Q10/Q50/Q90 projections verified.
- Match replay slider and Play functionality verified.
- Probability graph verified.

**Status: Completed**

---

## Overall Development Summary
Major capabilities delivered:

- Ball-by-ball match replay.
- Dockerized API.
- Automated CI testing.
- Deployed prediction API.
- First-innings quantile score predictions.
- Professional Tarazu Streamlit dashboard.
- Six-outcome what-if simulator.
- Model Trust and Transparency Panel.

**Latest automated test result: 21 passed, 0 failed.**


## Day 24 — Model Trust, Failure Analysis & Expert Review

### Objective

Move beyond displaying win probabilities and make the model's reliability, evaluation evidence, failure cases and limitations visible to users and developers.

### Work completed

#### 1. Trust evaluation notebook

Created `notebooks/trust.ipynb`.

- Loaded the held-out chase test dataset.
- Loaded the production XGBoost model and isotonic calibration artifacts.
- Generated predictions through the production `chase_curve` pipeline.
- Reproduced the evaluation metrics using the shared `cricketwp.eval` implementation.
- Compared production predictions against a constant training-frequency baseline.
- Generated quantile-based reliability bins.
- Exported the report to `artifacts/trust.json`.

An important evaluation detail was resolved: the saved test log loss uses probability clipping from the shared evaluation implementation. Using a different evaluation path initially produced a small discrepancy. Using the production evaluation implementation reproduced the saved metrics.

#### 2. Trust API

Updated `src/cricketwp/api/main.py`.

- Added trust-report loading from `artifacts/trust.json`.
- Added `GET /trust`.
- Added a 503 response when the report is unavailable.
- Verified the endpoint locally.

#### 3. Trust API tests

Updated `tests/test_api.py`.

Added tests covering:

- Trust endpoint response.
- Report version and test-set dimensions.
- Reliability-bin count.
- Metric availability and valid ranges.
- Model improvement over baseline for log loss, Brier score and ROC AUC.

#### 4. Tarazu Streamlit integration

Updated `app/streamlit_app.py`.

Added the third workspace mode: **Can I Trust This?**

Connected the panel to the `/trust` endpoint and implemented:

- Dynamic evaluation metrics.
- Baseline comparison.
- Reliability curve.
- Expandable reliability-bin table.
- Evaluation sample and match counts.
- Calibration and interpretation explanations.
- Known limitations and intended-use disclosures.

Verified all three workspace modes:

1. Live Scenario
2. Match Replay
3. Can I Trust This?

The dashboard's trust metrics and reliability visualizations were checked in the running application.

#### 5. Failure gallery

Created:

- `scripts/failures.py`
- `scripts/plot_failures.py`
- `docs/failure.md`
- Five visualizations under `docs/figs/`.

Reviewed five representative historical failures, including overconfident and underestimated predictions.

The investigation identified shortened-innings handling as a known issue in two examined cases.

#### 6. Expert review

Created:

- `scripts/expert_review.py`
- `docs/expert_review.md`

Analysed 49,388 late-innings states across 1,038 matches, excluding two known shortened-innings cases.

Investigated partnerships, wickets in hand and required-rate pressure.

The results are directional observational evidence, not causal conclusions.

### Important technical learning

**Evaluation must match the production prediction pipeline.**

The trust report uses calibrated predictions and terminal-state rules rather than reporting raw XGBoost metrics as though they were the deployed model's results.

Also, calibration and discrimination are different properties. ROC AUC measures ranking ability, while ECE evaluates probability calibration. A model can perform strongly on one and still have calibration weaknesses in particular probability ranges.

Ball-by-ball states are correlated within matches, so the number of states must not be confused with the number of independent matches.

### Issues and limitations carried forward

**F-01 — Shortened innings**

The 120-ball assumption remains unresolved in historical feature generation and related prediction logic.

The issue was documented and deferred during Day 24. It must be addressed before claiming validated support for shortened innings.

Other limitations include calibration differences across probability ranges, historical distribution dependence and the observational nature of the expert review.

### Verification

Executed:

`python -m pytest -v`

Result:

- 23 tests passed.
- 0 failures.
- 1 Starlette/httpx deprecation warning.

Verified the `/trust` endpoint and visually checked the three Streamlit workspace modes.

### Day 24 completion

Completed the trust-report generation, API integration, dashboard visualization, failure gallery and expert review.

The focus of Day 24 was model transparency and evidence-based evaluation, not claiming that all model limitations had been fixed.

### Next direction

Continue toward production-grade reliability by prioritising:

- Resolving F-01 shortened-innings handling.
- Deeper subgroup and match-level evaluation.
- Calibration monitoring across competitions and match situations.
- Further investigation of high-confidence prediction failures.
- Maintaining regression tests as prediction logic evolves.