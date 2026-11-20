import json

import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.predict import chase_curve

# --------------------------------------------------
# 1. Load locked test set and production artifacts
# --------------------------------------------------

te = pd.read_parquet("data/processed/chase_test.parquet")

clf = xgb.XGBClassifier()
clf.load_model("artifacts/chase_xgb.json")

with open("artifacts/chase_iso.json", encoding="utf-8") as f:
    iso = json.load(f)

curve = chase_curve(
    te,
    clf,
    np.asarray(iso["x"]),
    np.asarray(iso["y"]),
)

df = te.copy()
df["p_chase"] = curve["p_chase"].to_numpy()

# Exclude known shortened-innings cases affected by F-01.
known_affected = ["1488332", "1523004"]
df = df[~df["match_id"].astype(str).isin(known_affected)].copy()

# Focus on later innings.
df = df[df["balls_bowled"] >= 60].copy()

# --------------------------------------------------
# 2. Define measurable cricket hypotheses
# --------------------------------------------------

comparisons = {
    "Partnership runs": {
        "Higher / favourable": df["partnership_runs"] >= 30,
        "Lower / comparison": df["partnership_runs"] <= 10,
    },
    "Wickets in hand": {
        "Higher / favourable": df["wickets_in_hand"] >= 7,
        "Lower / comparison": df["wickets_in_hand"] <= 3,
    },
    "Required-rate pressure": {
        "Higher / pressure": df["rrr_minus_crr"] > 0,
        "Lower / no pressure": df["rrr_minus_crr"] <= 0,
    },
}

# --------------------------------------------------
# 3. Summarise predictions against actual outcomes
# --------------------------------------------------


def summarise(frame):
    frame = frame.dropna(subset=["p_chase", "label", "match_id"])

    if frame.empty:
        return None

    # Row-level statistics
    row_result = {
        "states": len(frame),
        "matches": frame["match_id"].nunique(),
        "mean_predicted": frame["p_chase"].mean(),
        "actual_win_rate": frame["label"].mean(),
        "brier": np.mean((frame["p_chase"] - frame["label"]) ** 2),
    }

    # Match-level averages prevent long matches from
    # dominating the comparison as heavily.
    per_match = frame.groupby("match_id").agg(
        predicted=("p_chase", "mean"),
        actual=("label", "mean"),
    )

    row_result["match_mean_predicted"] = per_match["predicted"].mean()
    row_result["match_mean_actual"] = per_match["actual"].mean()

    return row_result


print("\n" + "=" * 90)
print("DAY 24 — EXPERT REVIEW")
print("=" * 90)

print("Late-innings states:", len(df))
print("Unique matches:", df["match_id"].nunique())
print("Excluded known F-01 matches:", known_affected)

for belief, groups in comparisons.items():
    print("\n" + "-" * 90)
    print("BELIEF:", belief)
    print("-" * 90)

    for group_name, mask in groups.items():
        result = summarise(df[mask])

        print("\n", group_name)

        if result is None:
            print("No usable observations.")
            continue

        for key, value in result.items():
            if isinstance(value, (float, np.floating)):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")

print("\nReview complete.")
