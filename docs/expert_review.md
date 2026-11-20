# Expert Review — Day 24

## Objective

Evaluate three cricket hypotheses using calibrated predictions on the held-out chase test set.

The analysis covers **49,388 late-innings states across 1,038 matches**, excluding the two known shortened-innings cases affected by F-01.

## Results

### 1. Partnership runs

**Hypothesis:** Larger partnerships improve the chasing team's prospects.

| Metric | Partnership ≥30 | Partnership ≤10 |
|---|---:|---:|
| States | 11,983 | 22,676 |
| Predicted win rate | 59.49% | 28.45% |
| Actual win rate | 61.71% | 30.43% |

**Verdict: Supported.** Larger partnerships are associated with higher chase-win rates, and the model captures this direction.

### 2. Wickets in hand

**Hypothesis:** More wickets remaining improve the chances of a successful chase.

| Metric | ≥7 wickets | ≤3 wickets |
|---|---:|---:|
| States | 17,082 | 10,388 |
| Predicted win rate | 66.71% | 8.41% |
| Actual win rate | 69.40% | 8.82% |

**Verdict: Strongly supported.** The large difference in actual win rates is reflected in the model's predictions.

### 3. Required-rate pressure

**Hypothesis:** When the required run rate exceeds the current run rate, winning becomes harder.

| Metric | Under pressure | No pressure |
|---|---:|---:|
| States | 33,004 | 16,384 |
| Predicted win rate | 14.10% | 89.72% |
| Actual win rate | 16.80% | 92.58% |

**Verdict: Supported.** The model captures the strong relationship between required-rate pressure and chase outcomes.

## Conclusion

All three cricket hypotheses are supported directionally by the held-out data. The model reflects the expected relationships between partnership size, wickets in hand, required-rate pressure, and chase success.

## Limitations

- These are observational comparisons, not causal experiments.
- Multiple states from the same match are not independent.
- Groups may overlap and have different sample sizes.
- No statistical significance tests were performed.
- The findings do not establish calibration across every match situation.

**Overall:** The model demonstrates cricket-plausible directional behaviour across all three hypotheses, while further calibration and failure analysis remain necessary.