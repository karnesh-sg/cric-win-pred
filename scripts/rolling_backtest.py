import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.eval import evaluate
from cricketwp.features import (
    CHASE_FEATURES,
    MONO,
    add_venue_priors,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "interim"
NOTEBOOK_DIR = ROOT / "notebooks"
REPORT_DIR = ROOT / "reports"

YEARS = [2021, 2022, 2023, 2024, 2025]
N_TREES = 122

BASE_ELO = 1500
ELO_K = 20

SEED = 42


def add_date_snapshot_elo(matches, k=ELO_K, base=BASE_ELO):
    """
    Calculate pre-match Elo ratings using one shared rating snapshot
    for every match played on the same date.
    """
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "match_id"]).reset_index(drop=True)

    ratings = {}
    elo_bat_first = np.zeros(len(m))
    elo_chase = np.zeros(len(m))

    for date, day in m.groupby("date", sort=True):
        # All matches on this date use ratings from before this date.
        updates = {}

        for idx, row in day.iterrows():
            team_a = row["team_bat_first"]
            team_b = row["team_chase"]

            ra = ratings.get(team_a, base)
            rb = ratings.get(team_b, base)

            elo_bat_first[idx] = ra
            elo_chase[idx] = rb

            if pd.isna(row["chase_won"]):
                continue

            expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
            actual_a = 1 - row["chase_won"]

            change_a = k * (actual_a - expected_a)
            change_b = -change_a

            updates[team_a] = updates.get(team_a, 0.0) + change_a
            updates[team_b] = updates.get(team_b, 0.0) + change_b

        # Apply all results only after the entire date is processed.
        for team, change in updates.items():
            ratings[team] = ratings.get(team, base) + change

    m["elo_bat_first"] = elo_bat_first
    m["elo_chase"] = elo_chase
    m["elo_diff"] = m["elo_chase"] - m["elo_bat_first"]

    return m


def build_backtest_matches():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")

    # Recalculate venue priors using only earlier dates.
    matches = add_venue_priors(matches)

    # Recalculate Elo using one snapshot per date.
    matches = add_date_snapshot_elo(matches)

    matches["target_vs_venue"] = matches["first_total"] + 1 - matches["venue_avg_first"]

    return matches[
        [
            "match_id",
            "venue_avg_first",
            "venue_chase_rate",
            "elo_diff",
            "target_vs_venue",
        ]
    ]


def build_backtest_dataset():
    states = pd.read_parquet(DATA_DIR / "chase_final.parquet")

    match_features = build_backtest_matches()

    # Remove the old match-level features before merging.
    old_features = [
        "venue_avg_first",
        "venue_chase_rate",
        "elo_diff",
        "target_vs_venue",
    ]

    states = states.drop(columns=old_features)

    # Attach reconstructed historical features.
    states = states.merge(
        match_features,
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    states["date"] = pd.to_datetime(states["date_x"])

    return states


def validate_backtest_dataset(df):
    print("\n" + "=" * 60)
    print("BACKTEST DATA VALIDATION")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Matches: {df['match_id'].nunique():,}")

    print("\nDate range:")
    print(f"Start: {df['date'].min()}")
    print(f"End:   {df['date'].max()}")

    print("\nMissing values in model features:")
    print(df[CHASE_FEATURES].isna().sum())

    print("\nMatch counts by year:")
    print(df.groupby(df["date"].dt.year)["match_id"].nunique())

    # Essential integrity checks.
    assert df["match_id"].notna().all()
    assert df["date"].notna().all()
    assert df[CHASE_FEATURES].notna().all().all()

    assert set(YEARS).issubset(set(df["date"].dt.year.unique()))

    print("\nAll validation checks passed.")


def run_rolling_backtest(df):
    params_path = NOTEBOOK_DIR / "best_params.json"

    with open(params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    # Monotonic constraints must follow CHASE_FEATURES order.
    mono = "(" + ",".join(str(MONO.get(feature, 0)) for feature in CHASE_FEATURES) + ")"

    rows = []

    for year in YEARS:
        print(f"\nTraining and evaluating: {year}")

        train = df[df["date"] < f"{year}-01-01"].copy()

        test = df[(df["date"] >= f"{year}-01-01") & (df["date"] < f"{year + 1}-01-01")].copy()

        if train.empty or test.empty:
            print(f"Skipping {year}: empty train or test data.")
            continue

        model = xgb.XGBClassifier(
            n_estimators=N_TREES,
            tree_method="hist",
            monotone_constraints=mono,
            random_state=SEED,
            **best_params,
        )

        model.fit(
            train[CHASE_FEATURES],
            train["label"],
        )

        probabilities = model.predict_proba(test[CHASE_FEATURES])[:, 1]

        metrics = evaluate(
            test["label"].to_numpy(),
            probabilities,
        )

        result = {
            "year": year,
            "train_matches": train["match_id"].nunique(),
            "test_matches": test["match_id"].nunique(),
            "train_states": len(train),
            "test_states": len(test),
            **metrics,
        }

        rows.append(result)

        print(
            f"Test matches: {result['test_matches']:,} | "
            f"Log loss: {result['logloss']:.4f} | "
            f"ECE: {result['ece']:.4f}"
        )

    results = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "rolling_backtest.csv"
    results.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("ROLLING BACKTEST RESULTS")
    print("=" * 65)
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {output_path}")

    return results


def plot_rolling_backtest(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Log loss
    axes[0].plot(
        results["year"],
        results["logloss"],
        marker="o",
    )
    axes[0].set_title("Rolling Backtest — Log Loss")
    axes[0].set_xlabel("Test year")
    axes[0].set_ylabel("Log loss")
    axes[0].set_xticks(results["year"])
    axes[0].grid(alpha=0.3)

    # Expected calibration error
    axes[1].plot(
        results["year"],
        results["ece"],
        marker="o",
    )
    axes[1].set_title("Rolling Backtest — ECE")
    axes[1].set_xlabel("Test year")
    axes[1].set_ylabel("Expected calibration error")
    axes[1].set_xticks(results["year"])
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    output_path = REPORT_DIR / "rolling_backtest.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()

    print(f"Plot saved: {output_path}")


if __name__ == "__main__":
    df = build_backtest_dataset()
    validate_backtest_dataset(df)

    results = run_rolling_backtest(df)
    plot_rolling_backtest(results)
