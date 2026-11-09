# Model Card — Cricket Chase Win Probability Model

## 1. Model Overview

**Model name:** Cricket Chase Win Probability Model  
**Version:** 1.0.0  
**Model type:** XGBoost binary classification with isotonic probability calibration  
**Task:** Estimate the probability that the chasing team wins a men's T20 cricket match at a given point during the second innings.

The model produces a calibrated chase-win probability using the current match state and historical match information.

## 2. Intended Use

- Ball-by-ball cricket win probability analysis.
- Cricket analytics dashboards and educational demonstrations.
- Exploring how match conditions influence the likelihood of a successful chase.

This model is intended for analytical and educational use. It should not be treated as a guaranteed match outcome or used as the sole basis for betting or financial decisions.

## 3. Dataset

**Source:** Cricsheet ball-by-ball match data.  
**Scope:** Men's T20 matches only, including IPL and T20I data used in the project.

The data contains historical match outcomes and ball-by-ball match states. The target label is whether the chasing team eventually won the match.

The model uses a time-based train, validation, and test design. The test set represents a later period than the training data.

## 4. Model Inputs

The chase model uses 16 features, including match-state and historical context such as:

- Runs needed and balls remaining.
- Wickets in hand.
- Required run rate and current run rate relationship.
- Venue-level historical scoring and chase priors.
- Team Elo rating difference.

The complete feature list is stored in `artifacts/meta.json`.

## 5. Performance

Final evaluation on the held-out test set:

| Metric | Result |
|---|---:|
| Log Loss | 0.3217 |
| Brier Score | 0.1029 |
| ROC-AUC | 0.9377 |
| Expected Calibration Error (ECE) | 0.0319 |

The model demonstrates strong discrimination between successful and unsuccessful chases on the test data. Calibration metrics indicate that predicted probabilities are reasonably aligned with observed outcomes, though they are not perfect.

## 6. Limitations and Known Risks

- **Competition scope:** Men's T20 matches only. Performance is not established for women's cricket, ODI, or Test cricket.
- **Player quality:** The model does not explicitly account for individual batter or bowler quality.
- **Missing match conditions:** Toss decisions, dew, pitch characteristics, and other contextual conditions are not explicitly modeled.
- **Calibration drift:** Changes in scoring rates, playing conditions, rules, and team strategies may cause probability calibration to drift over time.
- **Test-period replay:** Historical matches from the test period are replayed ball by ball for evaluation. This is retrospective evaluation, not a live prospective trial.
- **Historical data limitations:** The model can only learn patterns represented in its training data. Rare match situations may produce less reliable estimates.
- **Probability interpretation:** A predicted probability is an estimate, not a guarantee of the final outcome.

## 7. Reproducibility

The model bundle contains:

- `chase_xgb.json` — trained XGBoost model.
- `chase_iso.json` — isotonic calibration mapping.
- `lookups.json` — venue priors, Elo ratings, and defaults.
- `meta.json` — model version, feature list, and test metrics.

**Model version:** 1.0.0  
**Status:** Evaluated and packaged; intended for further integration and monitoring.

##First-innings score projection: 
Three XGBoost quantile models (10th, 50th, and 90th percentiles) predict a final-score range from first-innings states. Validation median MAE: 19.50 runs; 10th–90th percentile coverage: 71.96%. Models are saved as `artifacts/score_q10.json`, `artifacts/score_q50.json`, and `artifacts/score_q90.json`.

## Batting-First Win Probability Model Card

### Model Overview
This model estimates the probability that the team batting first wins a T20 match based on the current innings state and contextual features.

### Model Type
- XGBoost Classifier
- Isotonic Regression calibration

### Features
- Runs
- Wickets in hand
- Balls left
- Current run rate
- Runs in the last 12 balls
- Wickets in the last 12 balls
- Dot-ball percentage in the last 12 balls
- Partnership runs
- Venue average first-innings score
- Venue chase rate
- Elo difference
- Innings phase

### Intended Use
- Live batting-first win probability estimation
- Cricket analytics and match-state visualization
- Identifying influential match-state factors

### Evaluation
Validation metrics:
- Logloss: 0.5601
- Brier score: 0.1906
- AUC: 0.7781

Test performance was also evaluated across innings phases. Model discrimination improved as the innings progressed.

### Limitations
- Calibration was fitted on the validation set; its near-zero ECE may be optimistic.
- Predictions from innings-state rows are correlated within matches.
- Cross-model consistency is not equivalent to prediction accuracy.
- Venue and Elo features may be affected by data availability and historical coverage.
- Performance may vary by venue, match format, and innings phase.

### Artifacts
- `artifacts/first_xgb.json`
- `artifacts/first_iso.json`
