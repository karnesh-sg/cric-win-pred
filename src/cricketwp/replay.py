from cricketwp.predict import chase_curve
from pathlib import Path

import json
import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.features import CHASE_FEATURES, FIRST_FEATURES


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim"
ARTIFACTS = ROOT / "artifacts"

first = pd.read_parquet(DATA / "first_final.parquet")
chase = pd.read_parquet(DATA / "chase_final.parquet")

print("First innings:", first.shape)
print("Chase innings:", chase.shape)

print("First columns:")
print(first.columns.tolist())

print("Chase columns:")
print(chase.columns.tolist())


# Keep matches from 2025 onward
first["date_x"] = pd.to_datetime(first["date_x"])
chase["date_x"] = pd.to_datetime(chase["date_x"])

test_ids = first.loc[
    first["date_x"] >= "2025-01-01",
    "match_id",
].unique()

first_te = first[first["match_id"].isin(test_ids)].copy()
chase_te = chase[chase["match_id"].isin(test_ids)].copy()

print("Test matches:", len(test_ids))
print("First innings test rows:", first_te.shape)
print("Chase innings test rows:", chase_te.shape)

print("First test date range:", first_te["date_x"].min(), "to", first_te["date_x"].max())
print("Chase test date range:", chase_te["date_x"].min(), "to", chase_te["date_x"].max())


# Load batting-first classifier
clf1 = xgb.XGBClassifier()
clf1.load_model(str(ARTIFACTS / "first_xgb.json"))

# Load isotonic calibration values
iso_f = json.loads((ARTIFACTS / "first_iso.json").read_text())
ix = np.array(iso_f["x"])
iy = np.array(iso_f["y"])

# Load score quantile models
q10_model = xgb.XGBRegressor()
q10_model.load_model(str(ARTIFACTS / "score_q10.json"))

q50_model = xgb.XGBRegressor()
q50_model.load_model(str(ARTIFACTS / "score_q50.json"))

q90_model = xgb.XGBRegressor()
q90_model.load_model(str(ARTIFACTS / "score_q90.json"))

score_models = [q10_model, q50_model, q90_model]


def first_curve(df):
    # Select features in the exact training order
    X = df[FIRST_FEATURES]

    # Predict score quantiles
    q = np.sort(
        np.column_stack([model.predict(X) for model in score_models]),
        axis=1,
    )

    # Predict raw batting-first win probability
    raw = clf1.predict_proba(X)[:, 1]

    # Calibrate probability
    p = np.interp(raw, ix, iy)

    return pd.DataFrame(
        {
            "p_bat_first_win": p,
            "q10": q[:, 0],
            "q50": q[:, 1],
            "q90": q[:, 2],
        },
        index=df.index,
    )


# Load batting-first classifier
clf1 = xgb.XGBClassifier()
clf1.load_model(str(ARTIFACTS / "first_xgb.json"))

# Load chase classifier
clf = xgb.XGBClassifier()
clf.load_model(str(ARTIFACTS / "chase_xgb.json"))

# Load isotonic calibration data
iso_f = json.loads((ARTIFACTS / "first_iso.json").read_text())
iso_c = json.loads((ARTIFACTS / "chase_iso.json").read_text())

ix, iy = iso_f["x"], iso_f["y"]
cx, cy = iso_c["x"], iso_c["y"]

# Load score quantile models
score_models = []

for name in ["score_q10.json", "score_q50.json", "score_q90.json"]:
    model = xgb.XGBRegressor()
    model.load_model(str(ARTIFACTS / name))
    score_models.append(model)

print("Models and calibrators loaded successfully.")


def first_curve(df):
    X = df[FIRST_FEATURES]

    q = np.sort(
        np.column_stack([model.predict(X) for model in score_models]),
        axis=1,
    )

    raw = clf1.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "p_bat_first_win": np.interp(raw, ix, iy),
            "q10": q[:, 0],
            "q50": q[:, 1],
            "q90": q[:, 2],
        },
        index=df.index,
    )


sample_id = first_te["match_id"].iloc[0]

sample = first_te[first_te["match_id"] == sample_id]

curve = first_curve(sample)

print("Sample match:", sample_id)
print("Input shape:", sample.shape)
print("Curve shape:", curve.shape)
print(curve.head())

# Build replay data for every test-period match
parts = []

for mid, f in first_te.groupby("match_id"):
    # Batting-first innings probability curve
    a = f.join(first_curve(f))

    # Chase innings for the same match
    c_ = chase_te[chase_te["match_id"] == mid]

    # Chase probability curve
    cc = chase_curve(c_, clf, cx, cy)

    # Convert chase probability to batting-first win probability
    b = c_.assign(p_bat_first_win=1 - cc["p_chase"].values)

    parts += [a, b]

# Combine innings into a single DataFrame
rp = pd.concat(parts)

# Create flags
rp["is_wicket"] = (rp["is_wicket"] > 0).astype(int)

rp["is_six"] = (rp["runs_batter"] == 6).astype(int)
rp["is_four"] = (rp["runs_batter"] == 4).astype(int)

# Rename match date
rp["date"] = rp["date_x"]

# Select the required columns
cols = [
    "match_id",
    "date",
    "venue",
    "innings",
    "batting_team",
    "bowling_team",
    "balls_bowled",
    "runs",
    "wkts",
    "p_bat_first_win",
    "q10",
    "q50",
    "q90",
    "is_wicket",
    "is_six",
    "is_four",
]

# Round probabilities and quantiles
replay_out = rp[cols].copy()

numeric_cols = replay_out.select_dtypes(include="number").columns
replay_out[numeric_cols] = replay_out[numeric_cols].round(4)

replay_out.to_csv(
    ARTIFACTS / "replay.csv.gz",
    index=False,
)

print("Replay shape:", rp[cols].shape)
print("Replay matches:", rp["match_id"].nunique())
print("Replay file created:", ARTIFACTS / "replay.csv.gz")
print(
    "Replay file size (MB):",
    round((ARTIFACTS / "replay.csv.gz").stat().st_size / (1024 * 1024), 2),
)


# Validate replay artifact
replay = pd.read_csv(ARTIFACTS / "replay.csv.gz")

print("\n--- Replay Validation ---")

print("Probability min:", replay["p_bat_first_win"].min())
print("Probability max:", replay["p_bat_first_win"].max())

print(
    "Invalid probabilities:",
    ((replay["p_bat_first_win"] < 0) | (replay["p_bat_first_win"] > 1)).sum(),
)

# Quantile ordering for first innings only
first_replay = replay[replay["innings"] == 1]

print(
    "Quantile ordering violations:",
    (
        (first_replay["q10"] > first_replay["q50"]) | (first_replay["q50"] > first_replay["q90"])
    ).sum(),
)

print("Missing probabilities:", replay["p_bat_first_win"].isna().sum())

print("Unique matches:", replay["match_id"].nunique())

print(
    "Flag values:",
    replay[["is_wicket", "is_six", "is_four"]].stack().unique(),
)
