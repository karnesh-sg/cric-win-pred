# Feature Drift Report

Comparison: Training data (2005–2023) vs 2026

- Training observations: 302,180
- Latest-season observations: 57,026

## Population Stability Index

| Feature | PSI | Status |
|---|---:|---|
| venue_avg_first | 0.5406 | Investigate |
| target | 0.1965 | Watch |
| target_vs_venue | 0.1736 | Watch |
| venue_chase_rate | 0.1575 | Watch |
| elo_diff | 0.1398 | Watch |
| crr | 0.1081 | Watch |
| rrr | 0.0772 | Stable |
| rrr_minus_crr | 0.0769 | Stable |
| runs_needed | 0.0297 | Stable |
| dot_pct_last12 | 0.0183 | Stable |
| runs_last12 | 0.0142 | Stable |
| wickets_in_hand | 0.0079 | Stable |
| balls_left | 0.0051 | Stable |
| partnership_runs | 0.0049 | Stable |
| wkts_last12 | 0.0045 | Stable |
| phase | 0.0044 | Stable |

## Interpretation

- PSI < 0.10: Stable
- PSI 0.10–0.25: Watch
- PSI > 0.25: Investigate

PSI measures feature-distribution changes, not model accuracy.

## Latest-season prediction evaluation

| Metric | Value |
|---|---:|
| Log loss | 0.308828 |
| Brier score | 0.098535 |
| AUC | 0.942970 |
| ECE | 0.032474 |
| Mean predicted probability | 0.436906 |
| Actual win rate | 0.464911 |
| Probability gap | -0.028005 |

The probability gap is mean predicted probability minus actual win rate.
A value close to zero indicates good aggregate calibration.

These metrics are calculated over state rows, not independent matches.
Multiple states from the same match are correlated, so this is a
descriptive monitoring evaluation rather than a match-level estimate.
