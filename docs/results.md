
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