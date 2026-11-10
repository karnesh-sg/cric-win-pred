
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