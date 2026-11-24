# Cricket Win Prediction — Runbook

## 1. Environment setup

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
pip install -e .
```

## 2. Run the complete ML pipeline

The pipeline executes five stages:

1. Parse raw Cricsheet JSON files into ball-level data.
2. Build match metadata and innings states.
3. Generate features and chronological train/validation/test splits.
4. Train and calibrate the prediction models.
5. Package models, lookups and metadata.

Run from the repository root:

```powershell
@'
from pathlib import Path
from cricketwp.pipeline import run_pipeline

result = run_pipeline(
    raw_dir=Path("data/raw"),
    workspace=Path("data/staging/h3_full_run"),
)

print("\nDATA FINGERPRINT:")
print(result["stage5"]["data_fingerprint_sha256"])

print("\nTEST METRICS:")
print(result["stage5"]["test_metrics"])
'@ | python -
```

**Important:** Each run requires a new workspace directory. Existing workspaces are not overwritten.

The pipeline writes its packaged artifacts to:

```text
data/staging/h3_full_run/package/
```

Production artifacts are not modified automatically.

## 3. Reproducibility verification

Training uses `random_state=42` for all XGBoost models.

To verify reproducibility:

1. Run the complete pipeline twice using separate staging workspaces.
2. Compare the data fingerprints.
3. Compare test metrics to six decimal places.
4. Confirm that the training summaries and model outputs are consistent.

The data fingerprint is calculated from the six chronological dataset splits.

## 4. Two-fixture smoke test

Run:

```powershell
python -m pytest tests/test_smoke.py -v
```

This verifies JSON parsing, raw delivery totals, ball dataset generation and match-state generation using the two fixtures in `tests/fixtures/`.

## 5. Full test suite

Run:

```powershell
python -m pytest tests/ -v
```

Confirm all tests pass before considering a release.

## 6. Production promotion

Pipeline execution writes to staging only.

Before promoting a package:

- Verify its metadata and data fingerprint.
- Review held-out test metrics.
- Confirm artifact compatibility with `Predictor`.
- Back up the current production artifacts.
- Promote only after explicit approval.

Never overwrite `artifacts/` or `data/processed/` during routine pipeline testing.

## 7. Model evaluation notes

The q10–q90 score interval currently has approximately 73% empirical coverage on the held-out test set, below its nominal 80% target. Treat these intervals as estimates, not calibrated 80% prediction intervals.