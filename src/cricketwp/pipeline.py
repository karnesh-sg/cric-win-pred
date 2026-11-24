import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.data.build_balls import build_balls
from cricketwp.eval import evaluate
from cricketwp.features import (
    CHASE_FEATURES,
    FIRST_FEATURES,
    add_elo,
    add_venue_priors,
)
from cricketwp.train import train_models

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}


def build_match_states(
    balls_path: Path,
    output_dir: Path,
) -> dict:
    """Build match metadata and first-innings/chase states."""

    balls_path = Path(balls_path)
    output_dir = Path(output_dir)

    b = pd.read_parquet(balls_path)

    # Normalize team names and venues.
    for col in ["batting_team", "bowling_team", "winner"]:
        b[col] = b[col].replace(TEAM_MAP)

    b["venue"] = b["venue"].str.split(",").str[0].str.strip()

    # Calculate innings totals.
    totals = b.groupby(["match_id", "innings"])["runs_total"].sum().unstack()

    # Build match metadata.
    m = b.drop_duplicates("match_id")[
        ["match_id", "date", "venue", "winner", "result", "method"]
    ].set_index("match_id")

    first_innings = b[b["innings"] == 1].drop_duplicates("match_id").set_index("match_id")

    m["team_bat_first"] = first_innings["batting_team"]
    m["team_chase"] = first_innings["bowling_team"]

    m["first_total"] = totals[1]
    m["second_total"] = totals[2]
    m["chase_won"] = (m["winner"] == m["team_chase"]).astype(int)

    # Apply the original notebook's sequential match exclusions.
    has_winner = m["winner"].notna()
    not_tie_or_no_result = m["result"].isna()
    not_dl = m["method"].isna()
    has_both_innings = m["first_total"].notna() & m["second_total"].notna()

    remaining = pd.Series(True, index=m.index)

    exclusions = {}

    exclusions["no_winner"] = int((remaining & ~has_winner).sum())
    remaining &= has_winner

    exclusions["tie_or_no_result"] = int((remaining & ~not_tie_or_no_result).sum())
    remaining &= not_tie_or_no_result

    exclusions["dl_method"] = int((remaining & ~not_dl).sum())
    remaining &= not_dl

    exclusions["missing_innings"] = int((remaining & ~has_both_innings).sum())
    remaining &= has_both_innings

    m = m[remaining].reset_index()

    # Build cumulative innings states.
    g = b.groupby(["match_id", "innings"], sort=False)

    b["runs"] = g["runs_total"].cumsum()
    b["wkts"] = g["is_wicket"].cumsum()
    b["balls_bowled"] = g["is_legal"].cumsum()

    # Retain legal deliveries only, after cumulative calculations.
    s = b[b["is_legal"] == 1].copy()

    s["balls_left"] = 120 - s["balls_bowled"]
    s["wickets_in_hand"] = 10 - s["wkts"]

    keep = [
        "match_id",
        "team_bat_first",
        "team_chase",
        "first_total",
        "chase_won",
    ]

    s = s.merge(
        m[keep],
        on="match_id",
        how="inner",
        validate="many_to_one",
    )

    # First-innings states.
    first = s[s["innings"] == 1].copy()
    first["final_total"] = first["first_total"]
    first["label_bat_first_won"] = 1 - first["chase_won"]

    # Chase states.
    chase = s[s["innings"] == 2].copy()
    chase["target"] = chase["first_total"] + 1
    chase["label"] = chase["chase_won"]

    # Write only to the caller-provided output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    m.to_parquet(output_dir / "matches.parquet", index=False)
    first.to_parquet(output_dir / "first_states.parquet", index=False)
    chase.to_parquet(output_dir / "chase_states.parquet", index=False)

    return {
        "matches": len(m),
        "first_rows": len(first),
        "chase_rows": len(chase),
        "exclusions": exclusions,
    }


def build_features_and_splits(
    balls_path: Path,
    states_dir: Path,
    output_dir: Path,
) -> dict:
    """Build engineered features and chronological train/validation/test splits."""

    import numpy as np

    from cricketwp.features import (
        add_elo,
        add_rolling_features,
        add_state_features,
        add_venue_priors,
    )

    balls_path = Path(balls_path)
    states_dir = Path(states_dir)
    output_dir = Path(output_dir)

    # ---------------------------------------------------------
    # 1. Load Stage 1 and Stage 2 datasets.
    # ---------------------------------------------------------
    balls = pd.read_parquet(balls_path)
    matches = pd.read_parquet(states_dir / "matches.parquet")
    first = pd.read_parquet(states_dir / "first_states.parquet")
    chase = pd.read_parquet(states_dir / "chase_states.parquet")

    # Standardize match IDs for reliable joins.
    balls["match_id"] = balls["match_id"].astype(str)
    matches["match_id"] = matches["match_id"].astype(str)
    first["match_id"] = first["match_id"].astype(str)
    chase["match_id"] = chase["match_id"].astype(str)

    # ---------------------------------------------------------
    # 2. Generate chronological match-level features.
    # ---------------------------------------------------------
    m_elo = add_elo(matches)

    # Reproduce the notebook's corrected venue-prior pass.
    m_venue = add_venue_priors(matches)
    original_venue = m_venue[["match_id", "venue_avg_first"]].rename(
        columns={"venue_avg_first": "venue_avg_first_original"}
    )

    venue_cols = ["venue_avg_first", "venue_chase_rate"]

    m_elo = m_elo.drop(columns=venue_cols, errors="ignore")
    m_elo = m_elo.merge(
        m_venue[["match_id"] + venue_cols],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    m_elo["elo_diff_bat"] = -m_elo["elo_diff"]

    match_features = [
        "match_id",
        "elo_diff",
        "elo_diff_bat",
    ]

    first = first.merge(
        m_elo[match_features],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    chase = chase.merge(
        m_elo[match_features],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # 3. Calculate rolling delivery features.
    # ---------------------------------------------------------
    balls_roll = balls.copy().reset_index(drop=True)
    balls_roll["_delivery_order"] = range(len(balls_roll))

    balls_roll = balls_roll.sort_values(
        ["match_id", "innings", "_delivery_order"],
        kind="stable",
    )

    balls_roll = add_rolling_features(balls_roll)

    balls_roll["balls_bowled"] = (
        balls_roll["is_legal"]
        .astype(int)
        .groupby([balls_roll["match_id"], balls_roll["innings"]])
        .cumsum()
    )

    rolling_cols = [
        "match_id",
        "innings",
        "balls_bowled",
        "runs_last12",
        "wkts_last12",
        "dot_pct_last12",
        "wkts_prev",
        "partnership_runs",
    ]

    rolling_states = balls_roll.loc[
        balls_roll["is_legal"].eq(True),
        rolling_cols,
    ].copy()

    # ---------------------------------------------------------
    # 4. Merge rolling features into innings states.
    # ---------------------------------------------------------
    first = first.merge(
        rolling_states[rolling_states["innings"] == 1].drop(columns="innings"),
        on=["match_id", "balls_bowled"],
        how="left",
        validate="one_to_one",
    )

    chase = chase.merge(
        rolling_states[rolling_states["innings"] == 2].drop(columns="innings"),
        on=["match_id", "balls_bowled"],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # 5. Generate state and derived features.
    # ---------------------------------------------------------
    first = add_state_features(first, chase=False)
    chase = add_state_features(chase, chase=True)

    first["wickets_in_hand"] = 10 - first["wkts"]
    chase["wickets_in_hand"] = 10 - chase["wkts"]

    chase = chase.merge(
        original_venue,
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    chase["target_vs_venue"] = chase["target"] - chase["venue_avg_first_original"]

    chase.drop(columns="venue_avg_first_original", inplace=True)

    # Restore the original state date as datetime.
    first["date"] = pd.to_datetime(first["date"])
    chase["date"] = pd.to_datetime(chase["date"])

    # ---------------------------------------------------------
    # 6. Append corrected venue priors at the end, matching
    #    the final notebook dataset column order.
    # ---------------------------------------------------------
    first = first.merge(
        m_elo[["match_id"] + venue_cols],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    chase = chase.merge(
        m_elo[["match_id"] + venue_cols],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # 6B. Match the established production dataset schemas.
    # ---------------------------------------------------------
    first_columns = [
        "match_id",
        "venue",
        "match_type",
        "gender",
        "winner",
        "result",
        "method",
        "innings",
        "over",
        "batting_team",
        "bowling_team",
        "runs_batter",
        "runs_total",
        "is_wicket",
        "is_legal",
        "runs",
        "wkts",
        "balls_bowled",
        "balls_left",
        "wickets_in_hand",
        "team_bat_first",
        "team_chase",
        "first_total",
        "chase_won",
        "final_total",
        "label_bat_first_won",
        "elo_diff",
        "elo_diff_bat",
        "runs_last12",
        "wkts_last12",
        "dot_pct_last12",
        "wkts_prev",
        "partnership_runs",
        "crr",
        "phase",
        "date",
        "venue_avg_first",
        "venue_chase_rate",
    ]

    chase_columns = [
        "match_id",
        "venue",
        "match_type",
        "gender",
        "winner",
        "result",
        "method",
        "innings",
        "over",
        "batting_team",
        "bowling_team",
        "runs_batter",
        "runs_total",
        "is_wicket",
        "is_legal",
        "runs",
        "wkts",
        "balls_bowled",
        "balls_left",
        "wickets_in_hand",
        "team_bat_first",
        "team_chase",
        "first_total",
        "chase_won",
        "target",
        "label",
        "elo_diff",
        "elo_diff_bat",
        "target_vs_venue",
        "runs_last12",
        "wkts_last12",
        "dot_pct_last12",
        "wkts_prev",
        "partnership_runs",
        "crr",
        "runs_needed",
        "rrr",
        "rrr_minus_crr",
        "phase",
        "date",
        "venue_avg_first",
        "venue_chase_rate",
    ]

    first = first[first_columns]
    chase = chase[chase_columns]

    # ---------------------------------------------------------
    # 7. Chronological splits.
    # ---------------------------------------------------------
    def split(df):
        train = df[df["date"] < "2024-01-01"].copy()

        validation = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")].copy()

        test = df[df["date"] >= "2025-01-01"].copy()

        return train, validation, test

    first_train, first_val, first_test = split(first)
    chase_train, chase_val, chase_test = split(chase)

    datasets = {
        "first_train": first_train,
        "first_val": first_val,
        "first_test": first_test,
        "chase_train": chase_train,
        "chase_val": chase_val,
        "chase_test": chase_test,
    }

    # ---------------------------------------------------------
    # 8. Validate and write to the caller-provided directory.
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in datasets.items():
        df.to_parquet(output_dir / f"{name}.parquet", index=False)

    def describe(df):
        return {
            "rows": len(df),
            "matches": int(df["match_id"].nunique()),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
        }

    return {
        "datasets": {name: describe(df) for name, df in datasets.items()},
        "first_missing_features": int(first.isna().sum().sum()),
        "chase_missing_features": int(chase.isna().sum().sum()),
        "rolling_rows": len(rolling_states),
    }


def build_lookups(matches: pd.DataFrame, chase_train: pd.DataFrame) -> dict:
    """Build runtime lookup tables using chronological historical data."""

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Venue priors: retain the latest historical prior for each venue.
    venue_history = add_venue_priors(matches)
    venue_history = venue_history.sort_values(["date", "match_id"])

    latest_venues = venue_history.groupby("venue", dropna=False).tail(1)

    venues = {
        str(row.venue): {
            "avg_first": float(row.venue_avg_first),
            "chase_rate": float(row.venue_chase_rate),
        }
        for row in latest_venues.itertuples()
        if pd.notna(row.venue)
    }

    # Global fallback values.
    prior_f = float(matches["first_total"].mean())
    prior_c = float(matches["chase_won"].mean())

    # Final Elo ratings, using the same chronological update rule.
    ratings = {}

    for row in matches.itertuples():
        a = row.team_bat_first
        b = row.team_chase

        ra = ratings.get(a, 1500.0)
        rb = ratings.get(b, 1500.0)

        if pd.isna(row.chase_won):
            continue

        expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
        result_a = 1 - row.chase_won

        ratings[a] = ra + 20 * (result_a - expected_a)
        ratings[b] = rb + 20 * ((1 - result_a) - (1 - expected_a))

    return {
        "venues": venues,
        "venue_default": {
            "avg_first": prior_f,
            "chase_rate": prior_c,
        },
        "elo": {str(team): float(rating) for team, rating in ratings.items()},
        "defaults": {"dot_pct_last12": float(chase_train["dot_pct_last12"].median())},
    }


def evaluate_test_models(data_dir: Path, model_dir: Path) -> dict:
    """Evaluate saved models on untouched chronological test datasets."""

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)

    chase_test = pd.read_parquet(data_dir / "chase_test.parquet")
    first_test = pd.read_parquet(data_dir / "first_test.parquet")

    # Chase classifier
    chase_model = xgb.XGBClassifier()
    chase_model.load_model(str(model_dir / "chase_xgb.json"))

    chase_iso = json.loads((model_dir / "chase_iso.json").read_text(encoding="utf-8"))

    chase_raw = chase_model.predict_proba(chase_test[CHASE_FEATURES])[:, 1]

    chase_prob = np.interp(chase_raw, chase_iso["x"], chase_iso["y"])

    # First-innings classifier
    first_model = xgb.XGBClassifier()
    first_model.load_model(str(model_dir / "first_xgb.json"))

    first_iso = json.loads((model_dir / "first_iso.json").read_text(encoding="utf-8"))

    first_raw = first_model.predict_proba(first_test[FIRST_FEATURES])[:, 1]

    first_prob = np.interp(first_raw, first_iso["x"], first_iso["y"])

    # First-innings score quantiles
    quantile_predictions = {}

    for q, filename in [
        (0.1, "score_q10.json"),
        (0.5, "score_q50.json"),
        (0.9, "score_q90.json"),
    ]:
        model = xgb.XGBRegressor()
        model.load_model(str(model_dir / filename))

        quantile_predictions[q] = model.predict(first_test[FIRST_FEATURES])

    actual_scores = first_test["final_total"].to_numpy()

    q10 = quantile_predictions[0.1]
    q50 = quantile_predictions[0.5]
    q90 = quantile_predictions[0.9]

    return {
        "chase": evaluate(chase_test["label"], chase_prob),
        "first": evaluate(first_test["label_bat_first_won"], first_prob),
        "score": {
            "q10_q90_coverage": float(np.mean((actual_scores >= q10) & (actual_scores <= q90))),
            "q50_mae": float(np.mean(np.abs(actual_scores - q50))),
        },
    }


def package_artifacts(
    data_dir: Path,
    states_dir: Path,
    model_dir: Path,
    output_dir: Path,
    training_summary: dict | None = None,
) -> dict:
    """Package verified models, lookups, and reproducibility metadata."""

    data_dir = Path(data_dir)
    states_dir = Path(states_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_files = [
        "chase_xgb.json",
        "chase_iso.json",
        "first_xgb.json",
        "first_iso.json",
        "score_q10.json",
        "score_q50.json",
        "score_q90.json",
    ]

    # Verify all model artifacts exist before packaging.
    for filename in model_files:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)

    # Copy trained model artifacts.
    for filename in model_files:
        shutil.copy2(model_dir / filename, output_dir / filename)

    # Build runtime lookup tables.
    matches = pd.read_parquet(states_dir / "matches.parquet")
    chase_train = pd.read_parquet(data_dir / "chase_train.parquet")

    lookups = build_lookups(matches, chase_train)

    (output_dir / "lookups.json").write_text(
        json.dumps(lookups, indent=2),
        encoding="utf-8",
    )

    # Evaluate only on the untouched test split.
    test_metrics = evaluate_test_models(data_dir, model_dir)

    # Fingerprint all six chronological split files.
    digest = hashlib.sha256()

    split_files = [
        f"{innings}_{split}.parquet"
        for innings in ("first", "chase")
        for split in ("train", "val", "test")
    ]

    for filename in sorted(split_files):
        path = data_dir / filename
        digest.update(filename.encode("utf-8"))
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

    # Capture source-control state.
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()

        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None
        git_dirty = None

    metadata = {
        "version": "1.1.0",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chase_features": CHASE_FEATURES,
        "first_features": FIRST_FEATURES,
        "test_metrics": test_metrics["chase"],
        "first_test_metrics": test_metrics["first"],
        "score_test_metrics": test_metrics["score"],
        "data_fingerprint_sha256": digest.hexdigest(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "library_versions": {
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
            "xgboost": package_version("xgboost"),
        },
        "training_summary": training_summary or {},
        "lookup_counts": {
            "venues": len(lookups["venues"]),
            "teams": len(lookups["elo"]),
        },
        "calibration": "isotonic",
    }

    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
        "test_metrics": test_metrics,
        "data_fingerprint_sha256": digest.hexdigest(),
    }


def run_pipeline(
    raw_dir: Path = Path("data/raw"),
    workspace: Path = Path("data/staging/h3_full_run"),
) -> dict:
    """Run all five ML pipeline stages in an isolated workspace."""

    raw_dir = Path(raw_dir)
    workspace = Path(workspace)

    if workspace.exists():
        raise FileExistsError(
            f"Workspace already exists: {workspace}. Choose a new staging directory."
        )

    workspace.mkdir(parents=True)

    balls_path = workspace / "interim" / "balls.parquet"
    states_dir = workspace / "states"
    features_dir = workspace / "features"
    models_dir = workspace / "models"
    package_dir = workspace / "package"

    print("\n========== STAGE 1: PARSE RAW DATA ==========")
    build_balls(raw_dir=raw_dir, output_path=balls_path)

    print("\n========== STAGE 2: BUILD MATCH STATES ==========")
    stage2 = build_match_states(balls_path=balls_path, output_dir=states_dir)

    print("\n========== STAGE 3: FEATURES AND SPLITS ==========")
    stage3 = build_features_and_splits(
        balls_path=balls_path,
        states_dir=states_dir,
        output_dir=features_dir,
    )

    print("\n========== STAGE 4: TRAIN MODELS ==========")
    stage4 = train_models(data_dir=features_dir, output_dir=models_dir)

    print("\n========== STAGE 5: PACKAGE ARTIFACTS ==========")
    stage5 = package_artifacts(
        data_dir=features_dir,
        states_dir=states_dir,
        model_dir=models_dir,
        output_dir=package_dir,
        training_summary=stage4,
    )

    summary = {
        "workspace": str(workspace),
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "stage5": stage5,
    }

    print("\n========== PIPELINE COMPLETE ==========")
    print("Package:", stage5["output_dir"])
    print("Fingerprint:", stage5["data_fingerprint_sha256"])

    return summary
