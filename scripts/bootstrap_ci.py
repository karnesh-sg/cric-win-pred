import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from cricketwp.predict import chase_curve

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "chase_train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "chase_test.parquet"

MODEL_PATH = ROOT / "artifacts" / "chase_xgb.json"
ISO_PATH = ROOT / "artifacts" / "chase_iso.json"

OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

N_BOOTSTRAPS = 500
RANDOM_SEED = 42


# ============================================================
# 2. CONSISTENT LOG-LOSS EVALUATION
# ============================================================


def clipped_logloss(y, probabilities):
    """
    Calculate log loss using the same clipping convention
    as cricketwp.eval.evaluate().
    """
    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    return log_loss(
        y,
        probabilities,
        labels=[0, 1],
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training matches:", train["match_id"].nunique())
print("Test matches:", test["match_id"].nunique())


# ============================================================
# 4. LOAD PRODUCTION MODEL AND CALIBRATION
# ============================================================

print("\nLoading production artifacts...")

clf = xgb.XGBClassifier()
clf.load_model(str(MODEL_PATH))

with open(ISO_PATH, encoding="utf-8") as f:
    iso = json.load(f)

iso_x = np.asarray(iso["x"])
iso_y = np.asarray(iso["y"])


# ============================================================
# 5. GENERATE PRODUCTION PREDICTIONS
# ============================================================

print("\nGenerating production predictions...")

curve = chase_curve(
    test,
    clf,
    iso_x,
    iso_y,
)

p_model = curve["p_chase"].to_numpy()
y = test["label"].to_numpy()

# Existing trust-report baseline.
baseline_probability = float(train["label"].mean())

p_baseline = np.full(
    len(test),
    baseline_probability,
)

assert len(p_model) == len(y) == len(p_baseline)
assert np.isfinite(p_model).all()
assert np.isfinite(p_baseline).all()
assert np.isin(y, [0, 1]).all()

# Verify the model's clipped log loss against the
# project's established evaluator.
model_ll = clipped_logloss(y, p_model)
baseline_ll = clipped_logloss(y, p_baseline)

observed_improvement = baseline_ll - model_ll

print("\nPOINT ESTIMATES")
print("------------------------------")
print(f"Baseline log loss: {baseline_ll:.6f}")
print(f"Model log loss:    {model_ll:.6f}")
print(f"Improvement:       {observed_improvement:.6f}")


# ============================================================
# 6. PREPARE MATCH-LEVEL CLUSTERS
# ============================================================

print("\nPreparing match-level clusters...")

match_groups = {
    match_id: np.asarray(indices)
    for match_id, indices in test.groupby(
        "match_id",
        sort=False,
    ).indices.items()
}

match_ids = np.asarray(list(match_groups.keys()))
n_matches = len(match_ids)

assert n_matches == test["match_id"].nunique()

print("Sampling units:", n_matches)
print("Total delivery states:", len(test))


# ============================================================
# 7. MATCH-LEVEL BOOTSTRAP
# ============================================================

print("\nRunning bootstrap...")

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_improvements = []
bootstrap_model_losses = []
bootstrap_baseline_losses = []

for iteration in range(N_BOOTSTRAPS):
    # Select match IDs with replacement.
    sampled_matches = rng.choice(
        match_ids,
        size=n_matches,
        replace=True,
    )

    # Include every state from each selected match.
    sampled_indices = np.concatenate([match_groups[match_id] for match_id in sampled_matches])

    y_sample = y[sampled_indices]
    model_sample = p_model[sampled_indices]
    baseline_sample = p_baseline[sampled_indices]

    model_loss = clipped_logloss(
        y_sample,
        model_sample,
    )

    baseline_loss = clipped_logloss(
        y_sample,
        baseline_sample,
    )

    improvement = baseline_loss - model_loss

    bootstrap_model_losses.append(model_loss)
    bootstrap_baseline_losses.append(baseline_loss)
    bootstrap_improvements.append(improvement)

    if (iteration + 1) % 100 == 0:
        print(f"Completed {iteration + 1}/{N_BOOTSTRAPS}")


# ============================================================
# 8. CALCULATE CONFIDENCE INTERVAL
# ============================================================

bootstrap_improvements = np.asarray(bootstrap_improvements)

ci_lower, ci_upper = np.percentile(
    bootstrap_improvements,
    [2.5, 97.5],
)

print("\n" + "=" * 55)
print("MATCH-LEVEL BOOTSTRAP RESULTS")
print("=" * 55)

print("Bootstrap repetitions:", N_BOOTSTRAPS)
print("Random seed:", RANDOM_SEED)
print("Test matches:", n_matches)

print(f"\nBaseline log loss: {baseline_ll:.6f}")
print(f"Model log loss: {model_ll:.6f}")
print(f"Observed improvement: {observed_improvement:.6f}")

print("\n95% CONFIDENCE INTERVAL")
print(f"Lower bound: {ci_lower:.6f}")
print(f"Upper bound: {ci_upper:.6f}")


# ============================================================
# 9. INTERPRETATION
# ============================================================

if ci_lower > 0:
    conclusion = (
        "The entire confidence interval is above zero. "
        "The model's improvement over the baseline is "
        "supported by this match-level bootstrap."
    )

elif ci_upper < 0:
    conclusion = (
        "The entire confidence interval is below zero. "
        "The baseline performs better in this comparison."
    )

else:
    conclusion = (
        "The confidence interval includes zero. "
        "The bootstrap does not establish a clear "
        "improvement over the baseline."
    )

print("\nINTERPRETATION")
print(conclusion)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results = {
    "method": "match-level cluster bootstrap",
    "bootstrap_repetitions": N_BOOTSTRAPS,
    "random_seed": RANDOM_SEED,
    "test_matches": int(n_matches),
    "test_states": len(test),
    "baseline_probability": baseline_probability,
    "baseline_logloss": float(baseline_ll),
    "model_logloss": float(model_ll),
    "observed_improvement": float(observed_improvement),
    "confidence_level": 0.95,
    "ci_lower": float(ci_lower),
    "ci_upper": float(ci_upper),
    "conclusion": conclusion,
    "evaluation": ("Log loss with probabilities clipped to [1e-6, 1-1e-6]"),
}

output_path = OUTPUT_DIR / "bootstrap_ci.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nSaved:", output_path)
print("\nBootstrap experiment completed.")
