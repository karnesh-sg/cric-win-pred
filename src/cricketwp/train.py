import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.isotonic import IsotonicRegression

from cricketwp.features import (
    CHASE_FEATURES,
    FIRST_FEATURES,
    MONO,
)


FIRST_MONO = {"runs": 1, "wickets_in_hand": 1}
CHASE_MONO = MONO


def build_monotone_constraints(features, constraints):
    return "(" + ",".join(str(constraints.get(feature, 0)) for feature in features) + ")"


def train_models(data_dir: Path, output_dir: Path) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training and validation datasets.
    chase_train = pd.read_parquet(data_dir / "chase_train.parquet")
    chase_val = pd.read_parquet(data_dir / "chase_val.parquet")

    first_train = pd.read_parquet(data_dir / "first_train.parquet")
    first_val = pd.read_parquet(data_dir / "first_val.parquet")

    # --------------------------------------------------
    # 1. Chase win-probability classifier
    # --------------------------------------------------

    chase_mono = build_monotone_constraints(CHASE_FEATURES, CHASE_MONO)

    chase_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        monotone_constraints=chase_mono,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )

    chase_model.fit(
        chase_train[CHASE_FEATURES],
        chase_train["label"],
        eval_set=[
            (
                chase_train[CHASE_FEATURES],
                chase_train["label"],
            ),
            (
                chase_val[CHASE_FEATURES],
                chase_val["label"],
            ),
        ],
        verbose=False,
    )

    # Calibrate chase probabilities.
    chase_raw = chase_model.predict_proba(chase_val[CHASE_FEATURES])[:, 1]

    chase_iso = IsotonicRegression(out_of_bounds="clip")
    chase_iso.fit(
        chase_raw,
        chase_val["label"],
    )

    chase_model.save_model(str(output_dir / "chase_xgb.json"))

    (output_dir / "chase_iso.json").write_text(
        json.dumps(
            {
                "x": chase_iso.X_thresholds_.tolist(),
                "y": chase_iso.y_thresholds_.tolist(),
            }
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 2. First-innings batting-first classifier
    # --------------------------------------------------

    first_mono = build_monotone_constraints(FIRST_FEATURES, FIRST_MONO)

    first_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        monotone_constraints=first_mono,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )

    first_model.fit(
        first_train[FIRST_FEATURES],
        first_train["label_bat_first_won"],
        eval_set=[
            (
                first_train[FIRST_FEATURES],
                first_train["label_bat_first_won"],
            ),
            (
                first_val[FIRST_FEATURES],
                first_val["label_bat_first_won"],
            ),
        ],
        verbose=False,
    )

    # Calibrate first-innings probabilities.
    first_raw = first_model.predict_proba(first_val[FIRST_FEATURES])[:, 1]

    first_iso = IsotonicRegression(out_of_bounds="clip")
    first_iso.fit(
        first_raw,
        first_val["label_bat_first_won"],
    )

    first_model.save_model(str(output_dir / "first_xgb.json"))

    (output_dir / "first_iso.json").write_text(
        json.dumps(
            {
                "x": first_iso.X_thresholds_.tolist(),
                "y": first_iso.y_thresholds_.tolist(),
            }
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 3. First-innings score quantile models
    # --------------------------------------------------

    quantile_models = {}

    for q, name in [
        (0.1, "score_q10.json"),
        (0.5, "score_q50.json"),
        (0.9, "score_q90.json"),
    ]:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=50,
        )

        model.fit(
            first_train[FIRST_FEATURES],
            first_train["final_total"],
            eval_set=[
                (
                    first_val[FIRST_FEATURES],
                    first_val["final_total"],
                )
            ],
            verbose=False,
        )

        model.save_model(str(output_dir / name))

        quantile_models[q] = model

    # --------------------------------------------------
    # 4. Training summary
    # --------------------------------------------------

    return {
        "chase_best_iteration": chase_model.best_iteration,
        "first_best_iteration": first_model.best_iteration,
        "score_best_iterations": {
            str(q): model.best_iteration for q, model in quantile_models.items()
        },
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
    }
