
# Cricket Win Probability — Model Results

## Validation Results

Experiment: `chase-winprob`

Validation split: 2024 (chronological)

| Model | Features | Log Loss | Brier Score | AUC | ECE |
|---|---|---:|---:|---:|---:|
| Baseline 0 — Training Average | Constant probability (0.491432) | 0.6910215 | 0.2489373 | 0.5000000 | 0.0577356 |
| Baseline 1 — Logistic Regression | runs_needed, balls_left, wickets_in_hand, rrr | 0.3801758 | 0.1242875 | 0.9095748 | 0.0438484 |

## Baseline Details

### Baseline 0 — Training Average
- Prediction: Training label average
- Training average: 0.4914322589185254
- A constant probability is predicted for every validation sample.

### Baseline 1 — Logistic Regression
- Algorithm: Logistic Regression
- Max iterations: 1000
- Features:
  - `runs_needed`
  - `balls_left`
  - `wickets_in_hand`
  - `rrr`
- Pipeline:
  - StandardScaler
  - LogisticRegression

## Evaluation Metrics

- **Log Loss:** Lower is better.
- **Brier Score:** Lower is better.
- **AUC:** Higher is better; 0.5 represents random ranking.
- **ECE:** Lower is better; measures calibration error.

## Notes

- Both baselines were logged in MLflow under the `chase-winprob` experiment.
- Results are from the chronological validation split.
- The locked test set (2025 onward) remains untouched.
- These baselines will serve as reference points for future models.


| XGBoost — Monotonic | CHASE_FEATURES | 0.3467304 | 0.1107270 | 0.9240892 | 0.0225579 |


### XGBoost — Monotonic Constraints

- Algorithm: XGBoost Classifier
- Maximum estimators: 2000
- Learning rate: 0.05
- Maximum depth: 5
- Subsample: 0.8
- Column sample by tree: 0.8
- Tree method: hist
- Monotonic constraints: Enabled
- Evaluation metric: Log Loss
- Early stopping rounds: 50
- Best iteration: 121

Validation metrics:
- Log Loss: 0.3467304
- Brier Score: 0.1107270
- AUC: 0.9240892
- ECE: 0.0225579

XGBoost outperformed both previous baselines on all four validation metrics.



### XGBoost Learning Curve — Overfitting Analysis

Training Log Loss continued to decrease after the best iteration (121), while validation Log Loss reached its minimum near iteration 121 and then flattened with a slight increase, indicating mild overfitting beyond this point. Early stopping at iteration 121 effectively limits this overfitting, so the current model configuration is retained without further complexity reduction.



## Match Group Cross-Validation

GroupKFold cross-validation was performed using 5 folds, keeping all ball-by-ball observations from each match within the same fold.

| Fold | Log Loss |
|---|---:|
| Fold 1 | 0.419818 |
| Fold 2 | 0.403592 |
| Fold 3 | 0.419695 |
| Fold 4 | 0.421771 |
| Fold 5 | 0.407632 |
| **Mean** | **0.414502** |
| **Standard Deviation** | **0.007407** |

The five folds produced relatively consistent Log Loss values, with a mean of 0.414502 and standard deviation of 0.007407. The grouped cross-validation Log Loss is higher than the chronological validation Log Loss of 0.346730, indicating that model performance varies across evaluation splits.

## Label-Shuffle Leakage Check

The XGBoost model was trained using randomly shuffled training labels and evaluated against the true validation labels.

| Metric | Shuffled-label result |
|---|---:|
| Log Loss | 0.692786 |
| Brier Score | 0.249806 |
| AUC | 0.472831 |
| ECE | 0.058829 |

The shuffled-label AUC of 0.472831 is close to the expected random-ranking value of 0.5. This provides no obvious evidence of leakage from the features into the target, although it does not rule out every possible leakage issue.

## Day 10 — Optuna Hyperparameter Tuning & Feature Ablation

### Optuna Hyperparameter Search

- Trials completed: 60
- Best validation Log Loss: 0.340546
- Day 9 validation Log Loss: 0.346730
- Relative Log Loss reduction: approximately 1.78%

Best parameters:

- max_depth: 6
- learning_rate: 0.126309
- subsample: 0.756896
- colsample_bytree: 0.720898
- min_child_weight: 50
- reg_lambda: 0.202514

### Feature Ablation

| Configuration | Validation Log Loss |
|---|---:|
| Full Model | 0.34055 |
| Without Momentum | 0.34930 |
| Without Venue | 0.35713 |
| Without Elo | 0.34818 |

### Conclusion

The full 16-feature model achieved the lowest validation Log Loss.

Venue features had the largest observed impact, followed by momentum and Elo features. Removing any of these groups worsened validation performance.

All feature groups are retained for subsequent model evaluation.

### SHAP Analysis

The SHAP summary plot shows that `rrr_minus_crr` and `rrr` are the most influential features, with higher values generally pushing predictions away from a batting-team win, reflecting increased chase difficulty. `wickets_in_hand` and `runs_needed` are also highly influential, with more wickets generally favouring the batting team and a larger number of runs remaining generally making the chase harder. Team strength (`elo_diff`) and venue-related features also influence predictions, indicating that the model uses both match-state information and contextual factors to estimate chase outcomes.


# Final Model Evaluation — 2026-11-07

## Overall Test Metrics

| Metric | Result |
|---|---:|
| Log Loss | 0.3217 |
| Brier Score | 0.1029 |
| ROC-AUC | 0.9377 |
| ECE | 0.0319 |

## Chase Phase Metrics

| Balls Bowled | Log Loss | Brier | AUC | ECE |
|---|---:|---:|---:|---:|
| (0, 36] | 0.4167 | 0.1362 | 0.8922 | 0.0418 |
| (36, 60] | 0.3387 | 0.1059 | 0.9337 | 0.0364 |
| (60, 90] | 0.2663 | 0.0842 | 0.9575 | 0.0298 |
| (90, 120] | 0.1845 | 0.0575 | 0.9792 | 0.0278 |

## Summary

The final calibrated model achieved a test ROC-AUC of **0.9377**. Predictive performance improved across later chase phases. The test set was reserved for final evaluation and not used for tuning.

## First-Innings Score Projection — Day 14

### Overall Validation Metrics

| Metric | Result |
|---|---:|
| Validation rows | 75,896 |
| Median MAE | 19.5004 runs |
| 10th–90th percentile coverage | 0.7196 |
| Coverage target | Approximately 0.80 |

### Accuracy by Stage of Innings

| Balls Bowled | Validation Rows | Median MAE (runs) | Coverage | Average Band Width (runs) |
|---|---:|---:|---:|---:|
| 0–36 | 23,430 | 30.02 | 0.683 | 76.21 |
| 36–60 | 15,563 | 22.14 | 0.688 | 58.46 |
| 60–90 | 19,102 | 15.60 | 0.725 | 42.16 |
| 90–120 | 17,798 | 7.54 | 0.790 | 22.63 |

### Interpretation
- Median MAE decreases as the innings progresses.
- Average prediction-band width decreases from 76.21 runs in the early stage to 22.63 runs in the late stage.
- Coverage improves from 68.3% to 79.0%.
- Overall coverage is 71.96%, below the approximate 80% target.
- The expected trend of decreasing uncertainty as more balls are bowled is observed.

### 60-Ball Prediction Plot
Saved to:

`docs/figs/first_innings_median_vs_actual_60_balls.png`

The scatter plot compares predicted median final totals against actual final totals for validation states at exactly 60 balls. The dashed diagonal represents perfect prediction.

### Hand-Test Results

| State | Low Prediction | Median Prediction | High Prediction |
|---|---:|---:|---:|
| 0 balls, high-scoring venue | 126.1 | 165.0 | 216.7 |
| 60/1 after 10 overs | 125.9 | 151.0 | 183.2 |
| 40/5 after 10 overs | 70.8 | 97.7 | 123.5 |
| 150/3 after 17 overs | 171.9 | 182.3 | 197.3 |
| 100/9 after 18 overs | 106.0 | 107.5 | 121.3 |

All five hand-tested score ranges were plausible and ordered correctly.

### Saved Models
- `artifacts/score_q10.json`
- `artifacts/score_q50.json`
- `artifacts/score_q90.json`

## Day 15 — Batting-First Win Probability Model

### Model Training
- Model: XGBoost Classifier
- Best iteration: 119
- Best validation logloss: 0.5683
- Learning rate: 0.05
- Maximum depth: 4
- Early stopping rounds: 50
- Monotonic constraints:
  - Runs: +1
  - Wickets in hand: +1

### Validation Results — Calibrated Model

| Metric | Value |
|---|---:|
| Logloss | 0.5601 |
| Brier Score | 0.1906 |
| AUC | 0.7781 |
| ECE | ≈ 0.0000 |

Note: Isotonic calibration was fitted and evaluated on the same validation set. The near-zero ECE should therefore not be interpreted as a generalization guarantee.

### Test Performance by Innings Phase

| Phase (balls bowled) | Rows | Logloss | Brier | AUC | ECE |
|---|---:|---:|---:|---:|---:|
| 0–36 | 40,764 | 0.6020 | 0.2075 | 0.7437 | 0.0491 |
| 36–60 | 27,038 | 0.5632 | 0.1904 | 0.7953 | 0.0647 |
| 60–90 | 33,205 | 0.5317 | 0.1787 | 0.8168 | 0.0580 |
| 90–120 | 31,035 | 0.5120 | 0.1710 | 0.8322 | 0.0494 |

### Interpretation
- Logloss and Brier score improve as the innings progresses.
- AUC improves across innings phases, suggesting stronger discrimination when more information is available.
- Calibration varies across phases and requires further evaluation.
- Innings-state rows are not independent match observations.

### Cross-Model Consistency

The batting-first model was compared against the chase model using the final first-innings state and the first chase state for each match. Predictions were paired by `match_id`.

| Metric | Value |
|---|---:|
| Matched test matches | 1,133 |
| Average consistency gap | 8.77 percentage points |
| Median consistency gap | 6.77 percentage points |
| 90th percentile gap | 18.05 percentage points |

The average consistency gap is a diagnostic of agreement between the two models. It is not a measure of accuracy.


# Day 16 — Results

## Completed
- Generated replay dataset for 1,133 matches (2025–2026).
- Implemented `GET /matches`.
- Implemented `GET /matches/{match_id}/replay`.
- Verified replay API responses.
- Added 5 API tests covering:
  - Probability sum to 1
  - Invalid input validation (HTTP 422)
  - Target reached rule
  - Target monotonicity
  - Wicket monotonicity

## Test Results
- **17 tests passed**
- 0 failures
- 1 Starlette deprecation warning

## Status
Day 16 completed. API tests are green.

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


## 1. Overall Results

During Days 18–23, the cricket win-probability project was extended with interactive replay, API deployment, first-innings score prediction, dashboard improvements, hypothetical delivery simulation and model transparency.

The application now combines trained ML models, a FastAPI backend and a Streamlit frontend.

### Major deliverables

| Feature | Result |
|---|---|
| Ball-by-ball match replay | Completed |
| Dockerized API | Completed |
| CI integration | Passed |
| API deployment | Verified |
| First-innings quantile predictions | Integrated |
| Tarazu dashboard | Implemented |
| What-if simulator | Integrated |
| Model Trust Panel | Verified |
| Automated tests | 21 passed |

---

## Day 18 — Match Replay Results

### Delivered
- Interactive ball-by-ball match replay.
- Match selection.
- Replay slider.
- Playback functionality.
- Win-probability graph.
- Wicket and six markers.
- Current-delivery indicator.

### Outcome

Users can inspect how the predicted probability changes as a match progresses.

The chart displays the batting-first team's win probability and the opposing team's complementary probability.

**Result: Replay functionality successfully integrated and manually verified.**

---

##  Day 19 — Docker and CI Results

### Delivered
- Dockerized FastAPI application.
- Successful Docker image build.
- Running API container.
- Health endpoint verification.
- Model metadata verification.
- Prediction endpoint testing.
- Swagger UI verification.
- GitHub Actions CI verification.

### Container details

| Item | Value |
|---|---|
| Container | `cric-win-pred-api` |
| Host port | `8001` |
| Existing container port | `8000` |
| Model version | `1.0.0` |

### Outcome

The prediction API can run inside a Docker container and its main endpoints were verified.

GitHub Actions completed successfully.

**Result: Docker and CI integration completed.**

---

## Day 20 — Deployment Results

### Delivered
- Deployed FastAPI prediction service.
- Verified deployed chase prediction endpoint.
- Verified deployed first-innings prediction endpoint.
- Checked production API responses.

### Production API

https://cric-win-pred.onrender.com

### Outcome

The prediction backend became accessible through a deployed API.

**Result: Deployment verified.**

---

##  Day 21 — First-Innings Prediction Results

### Delivered

Integrated trained quantile models into the dashboard.

| Quantile | Interpretation |
|---|---|
| Q10 | Lower estimated score |
| Q50 | Median estimated score |
| Q90 | Upper estimated score |

### Outcome

The dashboard now displays a predicted first-innings score range using trained quantile models instead of a manually calculated heuristic.

These estimates are conditional on the current match state.

**Result: First-innings score projection integrated and manually verified.**

---

##  Day 22 — Dashboard Results

### Delivered

Created the branded cricket analytics dashboard:

**Tarazu · तराज़ू · தராசு**

**CRICKET INTELLIGENCE PLATFORM**

Integrated dashboard functionality:

- Live match-state prediction.
- Win-probability visualization.
- What-if analysis interface.
- First-innings score projection.
- Ball-by-ball replay.
- Model feature explanations.
- Premium dark UI.

### Outcome

The frontend brings the project's prediction and analysis capabilities together in one application.

**Result: Dashboard integration and visual improvements completed.**

---

##  Day 23 — What-If Simulator Results

### Delivered

Implemented the `POST /what-if/chase` API.

The simulator evaluates six hypothetical next-ball outcomes.

| Outcome | Description |
|---|---|
| Dot | No run |
| 1 | One run |
| 2 | Two runs |
| 4 | Four runs |
| 6 | Six runs |
| Wicket | One wicket lost |

For each outcome, the API returns:

- Hypothetical batting-team win probability.
- Change from the current win probability.

The Streamlit dashboard displays these results as six interactive scenario cards.

### Outcome

The user can explore how different possible deliveries could change the predicted match outcome.

The cards dynamically update when the match state changes.

**Result: What-if simulator successfully integrated and verified.**

---

##  Day 23 — Model Trust and Transparency Results

### Chase model test metrics

The shared `meta.json` records the following held-out test results:

| Metric | Value |
|---|---:|
| ROC AUC | **0.9377** |
| Log Loss | **0.3217** |
| Brier Score | **0.1029** |
| Expected Calibration Error | **0.0319** |

### Model metadata

| Property | Value |
|---|---|
| Model version | `1.0.0` |
| Metadata date | `2026-11-09` |
| Calibration method | Isotonic |

### Interpretation

- ROC AUC measures the model's ability to distinguish between winning and losing outcomes.
- Log loss evaluates the quality of predicted probabilities.
- Brier score measures squared probability error.
- ECE measures the difference between predicted confidence and observed outcome frequency.

Lower log loss, Brier score and ECE are desirable. Higher ROC AUC is desirable.

**Important:** These are chase-model test metrics. First-innings test metrics are not recorded in the shared metadata file and are not claimed here.

### Trust Panel

The dashboard now explains:

- Model evaluation metrics.
- Calibration.
- Quantile predictions.
- TreeSHAP explanations.
- What-if assumptions.
- Known limitations.

**Result: Model Trust Panel visually verified.**

---

## 9. Final Testing Results

### Automated testing

Command:

`pytest -q`

<badge color="success">21 PASSED</badge>

| Test result | Value |
|---|---:|
| Passed | 21 |
| Failed | 0 |
| Runtime | 7.42 seconds |
| Warnings | 1 |

### Warning

One non-blocking deprecation warning:

Starlette currently warns that using `httpx` with `starlette.testclient` is deprecated and recommends installing `httpx2`.

This does not represent a failing test.

### Manual verification

| Component | Result |
|---|---|
| Live chase prediction | Verified |
| What-if simulator | Verified |
| Dynamic scenario updates | Verified |
| Model Trust Panel | Verified |
| First-innings Q10/Q50/Q90 | Verified |
| Replay slider | Verified |
| Replay playback | Verified |
| Probability graph | Verified |

---

## 10. Technical Limitations and Future Improvements

1. Record first-innings held-out evaluation metrics separately from chase-model metrics.
2. Continue monitoring model performance and calibration on future matches.
3. Investigate the Starlette/httpx deprecation warning and update dependencies compatibly.
4. Continue improving what-if simulation assumptions and validation.
5. Evaluate model performance across different match phases, venues and teams.
6. Continue improving production reliability, API monitoring and dashboard responsiveness.

---

## Final Status

The project now has an interactive replay system, deployed API, first-innings quantile predictions, professional dashboard, what-if simulator and model transparency interface.

**Automated tests: 21 passed, 0 failed.**

## Day 24 — Model Trust, Failure Analysis & Expert Review

### 1. Production model evaluation

Created a dedicated trust evaluation report using the production chase prediction pipeline.

**Held-out test set**

| Metric | Production model | Constant baseline |
|---|---:|---:|
| Log loss | 0.321717 | 0.692339 |
| Brier score | 0.102873 | 0.249596 |
| ROC AUC | 0.937680 | 0.500000 |
| ECE | 0.031884 | 0.019290 |

- Test states: 114,440
- Test matches: 1,133
- Model version: 1.0.0
- Evaluation date: 9 November 2026

The baseline predicts a constant probability equal to the training-set chase-win frequency.

The production model substantially improves log loss, Brier score and ROC AUC over this baseline. The baseline's lower ECE does not imply better predictive performance: a constant prediction has no discriminatory ability.

Metrics were reproduced using the shared evaluation implementation and production prediction pipeline, including isotonic calibration and terminal-state rules.

### 2. Reliability analysis

Generated a nine-bin quantile-based reliability report.

| Predicted probability | Observed win rate | States |
|---:|---:|---:|
| 0.005344 | 0.003003 | 22,976 |
| 0.074377 | 0.057724 | 11,555 |
| 0.166440 | 0.186331 | 13,417 |
| 0.302443 | 0.369729 | 9,296 |
| 0.469509 | 0.574116 | 12,440 |
| 0.704890 | 0.772608 | 13,712 |
| 0.857832 | 0.888943 | 8,158 |
| 0.960074 | 0.963926 | 13,112 |
| 0.997312 | 0.995703 | 9,774 |

The reliability analysis shows that calibration varies across probability ranges. In particular, the middle probability groups show underconfidence, where observed win frequencies exceed average predicted probabilities.

The number of bins is nine because duplicate probability boundaries reduce the number of distinct quantile groups.

### 3. Trust report and API

Created `artifacts/trust.json` containing:

- Model version and evaluation date.
- Production test metrics.
- Constant-baseline metrics and definition.
- Reliability-bin data.
- Number of test states and matches.
- Known evaluation limitations.

Added the `GET /trust` API endpoint to expose this report.

Added API tests for report availability, metric validity and comparison against the baseline.

### 4. Streamlit trust dashboard

Integrated the trust report into the Tarazu dashboard under **Can I Trust This?**

Implemented:

- API-backed performance metrics.
- Model-versus-baseline comparison.
- Reliability curve.
- Expandable reliability-bin table.
- Evaluation sample counts.
- Calibration explanation.
- First-innings quantile-model explanation.
- TreeSHAP explanation and interpretation caveats.
- What-if simulator assumptions.
- Known limitations and intended-use disclosure.

The Live Scenario, Match Replay and Can I Trust This? workspace modes were verified.

### 5. Failure analysis

Created a failure-analysis workflow using:

- `scripts/failures.py`
- `scripts/plot_failures.py`
- `docs/failure.md`
- `docs/figs/`

Generated five visualizations covering matches:

- 148