import json

import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.predict import chase_curve

# Load test data and model
te = pd.read_parquet("data/processed/chase_test.parquet")

clf = xgb.XGBClassifier()
clf.load_model("artifacts/chase_xgb.json")

with open("artifacts/chase_iso.json", encoding="utf-8") as f:
    iso = json.load(f)

# Reproduce calibrated production predictions
curve = chase_curve(
    te,
    clf,
    np.asarray(iso["x"]),
    np.asarray(iso["y"]),
)

t = te.copy()
t["p_chase"] = curve["p_chase"].to_numpy()

# Six candidates from the two rankings
candidate_ids = [
    1473468,
    1502748,
    1523004,
    1528306,
    1488332,
    1557533,
]

for match_id in candidate_ids:
    m = t[t["match_id"].astype(str) == str(match_id)].sort_values("balls_bowled").copy()

    if m.empty:
        print(f"\nMATCH NOT FOUND: {match_id}")
        continue

    m["p_winner"] = np.where(
        m["label"] == 1,
        m["p_chase"],
        1 - m["p_chase"],
    )

    post_pp = m[m["balls_bowled"] > 36].copy()

    if post_pp.empty:
        continue

    worst = post_pp.loc[post_pp["p_winner"].idxmin()]

    print("\n" + "=" * 100)
    print("MATCH:", match_id)
    print("=" * 100)

    print("Date:", worst["date"])
    print("Winner:", worst["winner"])
    print("Batting first:", worst["team_bat_first"])
    print("Chasing team:", worst["team_chase"])
    print("Label:", worst["label"])
    print("Chase won:", worst["chase_won"])
    print("First innings total:", worst["first_total"])
    print("Target:", worst["target"])

    print("\nWORST PREDICTION:")
    print(
        m.loc[
            [worst.name],
            [
                "balls_bowled",
                "runs",
                "wkts",
                "runs_needed",
                "balls_left",
                "wickets_in_hand",
                "p_chase",
                "p_winner",
            ],
        ].to_string(index=False)
    )

    print("\nFIRST 5 POST-POWERPLAY STATES:")
    print(
        m[m["balls_bowled"] > 36][
            [
                "balls_bowled",
                "runs",
                "wkts",
                "runs_needed",
                "balls_left",
                "p_chase",
                "p_winner",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nLAST 5 STATES:")
    print(
        m[
            [
                "balls_bowled",
                "runs",
                "wkts",
                "runs_needed",
                "balls_left",
                "p_chase",
                "p_winner",
            ]
        ]
        .tail(5)
        .to_string(index=False)
    )

    print("\nMATCH DATA CHECK:")
    print("Unique labels:", m["label"].unique())
    print("Unique winners:", m["winner"].unique())
    print("Unique chasing teams:", m["team_chase"].unique())
    print("Unique batting-first teams:", m["team_bat_first"].unique())
    print("Duplicate ball numbers:", m["balls_bowled"].duplicated().sum())
