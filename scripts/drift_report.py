from pathlib import Path

import numpy as np
import pandas as pd

from cricketwp.eval import evaluate
from cricketwp.predict import apply_rules
from cricketwp.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "chase_train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "chase_test.parquet"

OUTPUT_PATH = ROOT / "docs" / "drift_report.md"

CHASE_FEATURES = [
    "runs_needed",
    "balls_left",
    "wickets_in_hand",
    "crr",
    "rrr",
    "rrr_minus_crr",
    "target",
    "runs_last12",
    "wkts_last12",
    "dot_pct_last12",
    "partnership_runs",
    "venue_avg_first",
    "venue_chase_rate",
    "elo_diff",
    "target_vs_venue",
    "phase",
]

EPSILON = 0.0001


def calculate_psi(expected, actual, bins):
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.clip(expected_pct, EPSILON, None)
    actual_pct = np.clip(actual_pct, EPSILON, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def create_numeric_bins(training_values, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins + 1)

    bins = np.quantile(training_values, quantiles)
    bins = np.unique(bins)

    bins[0] = -np.inf
    bins[-1] = np.inf

    return bins


def calculate_categorical_psi(expected, actual, categories):
    expected_counts = (
        pd.Series(expected).value_counts().reindex(categories, fill_value=0).to_numpy()
    )

    actual_counts = pd.Series(actual).value_counts().reindex(categories, fill_value=0).to_numpy()

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.clip(expected_pct, EPSILON, None)
    actual_pct = np.clip(actual_pct, EPSILON, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    test["date"] = pd.to_datetime(test["date"])

    latest_year = test["date"].dt.year.max()

    latest = test[test["date"].dt.year == latest_year].copy()

    return train, latest, latest_year


def build_psi_report(train, latest):
    results = []

    for feature in CHASE_FEATURES:
        expected = train[feature].to_numpy()
        actual = latest[feature].to_numpy()

        if feature == "phase":
            categories = sorted(train[feature].unique())

            psi = calculate_categorical_psi(
                expected,
                actual,
                categories,
            )

        else:
            bins = create_numeric_bins(expected)

            psi = calculate_psi(
                expected,
                actual,
                bins,
            )

        if psi < 0.10:
            status = "Stable"
        elif psi <= 0.25:
            status = "Watch"
        else:
            status = "Investigate"

        results.append(
            {
                "feature": feature,
                "psi": psi,
                "status": status,
            }
        )

    return pd.DataFrame(results).sort_values(
        "psi",
        ascending=False,
    )


def save_psi_report(psi_df, train, latest, latest_year):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Feature Drift Report",
        "",
        f"Comparison: Training data (2005–2023) vs {latest_year}",
        "",
        f"- Training observations: {len(train):,}",
        f"- Latest-season observations: {len(latest):,}",
        "",
        "## Population Stability Index",
        "",
        "| Feature | PSI | Status |",
        "|---|---:|---|",
    ]

    for row in psi_df.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.psi:.4f} | {row.status} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- PSI < 0.10: Stable",
            "- PSI 0.10–0.25: Watch",
            "- PSI > 0.25: Investigate",
            "",
            "PSI measures feature-distribution changes, not model accuracy.",
            "",
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved drift report: {OUTPUT_PATH}")


def predict_latest_season(latest):
    predictor = Predictor(folder=str(ROOT / "artifacts"))

    X = latest[CHASE_FEATURES]

    # 1. Raw XGBoost probabilities
    raw_p = predictor.chase_model.predict_proba(X)[:, 1]

    # 2. Isotonic calibration
    iso = predictor.iso_c

    calibrated_p = np.interp(
        raw_p,
        iso["x"],
        iso["y"],
    )

    # 3. Apply cricket terminal-state rules
    probabilities = apply_rules(
        calibrated_p,
        runs_needed=latest["runs_needed"].to_numpy(),
        balls_left=latest["balls_left"].to_numpy(),
        wickets_in_hand=latest["wickets_in_hand"].to_numpy(),
    )

    return probabilities


def evaluate_latest_season(latest, probabilities):
    y_true = latest["label"].to_numpy()

    metrics = evaluate(y_true, probabilities)

    mean_predicted_probability = float(np.mean(probabilities))
    actual_win_rate = float(np.mean(y_true))

    results = {
        "logloss": metrics["logloss"],
        "brier": metrics["brier"],
        "auc": metrics["auc"],
        "ece": metrics["ece"],
        "mean_predicted_probability": mean_predicted_probability,
        "actual_win_rate": actual_win_rate,
        "probability_gap": (mean_predicted_probability - actual_win_rate),
    }

    return results


def append_calibration_report(results):
    lines = [
        "",
        "## Latest-season prediction evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Log loss | {results['logloss']:.6f} |",
        f"| Brier score | {results['brier']:.6f} |",
        f"| AUC | {results['auc']:.6f} |",
        f"| ECE | {results['ece']:.6f} |",
        f"| Mean predicted probability | {results['mean_predicted_probability']:.6f} |",
        f"| Actual win rate | {results['actual_win_rate']:.6f} |",
        f"| Probability gap | {results['probability_gap']:.6f} |",
        "",
        "The probability gap is mean predicted probability minus actual win rate.",
        "A value close to zero indicates good aggregate calibration.",
        "",
        "These metrics are calculated over state rows, not independent matches.",
        "Multiple states from the same match are correlated, so this is a",
        "descriptive monitoring evaluation rather than a match-level estimate.",
        "",
    ]

    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    train, latest, latest_year = load_data()

    print(f"Training rows: {len(train):,}")
    print(f"Latest season: {latest_year}")
    print(f"Latest-season rows: {len(latest):,}")

    psi_df = build_psi_report(train, latest)

    print("\nFEATURE DRIFT RESULTS")
    print(psi_df.to_string(index=False))

    save_psi_report(
        psi_df,
        train,
        latest,
        latest_year,
    )

    probabilities = predict_latest_season(latest)

    prediction_results = evaluate_latest_season(
        latest,
        probabilities,
    )

    print("\nLATEST-SEASON PREDICTION EVALUATION")

    for metric, value in prediction_results.items():
        print(f"{metric}: {value:.6f}")

    append_calibration_report(prediction_results)


if __name__ == "__main__":
    main()
