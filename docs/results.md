
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
