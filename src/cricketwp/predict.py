import numpy as np
from cricketwp.features import CHASE_FEATURES
import xgboost as xgb


def apply_rules(p, runs_needed, balls_left, wickets_in_hand):
    p = np.where(runs_needed <= 0, 1.0, p)

    p = np.where((balls_left <= 0) & (runs_needed > 0), 0.0, p)

    p = np.where((wickets_in_hand <= 0) & (runs_needed > 0), 0.0, p)

    return p


def chase_curve(df, clf, iso_x, iso_y):
    # Step 1: Get raw win probabilities from the model
    raw = clf.predict_proba(df[CHASE_FEATURES])[:, 1]

    # Step 2: Calibrate the raw probabilities
    p = np.interp(raw, iso_x, iso_y)

    # Step 3: Apply cricket-specific terminal rules
    p = apply_rules(p, df.runs_needed.values, df.balls_left.values, df.wickets_in_hand.values)

    # Step 4: Create the output DataFrame
    out = df[["balls_bowled"]].copy()
    out["p_chase"] = p

    # Step 5: Calculate Win Probability Added
    out["wpa"] = out.p_chase.diff().fillna(0)

    return out


import numpy as np
import xgboost as xgb

from cricketwp.features import CHASE_FEATURES


LABELS = {
    "runs_needed": "Runs needed",
    "balls_left": "Balls left",
    "wickets_in_hand": "Wickets in hand",
    "crr": "Current run rate",
    "rrr": "Required run rate",
    "rrr_minus_crr": "Required vs current run rate",
    "target": "Target score",
    "runs_last12": "Runs in last 2 overs",
    "wkts_last12": "Wickets in last 2 overs",
    "dot_pct_last12": "Dot-ball percentage in last 2 overs",
    "partnership_runs": "Partnership runs",
    "venue_avg_first": "Venue scoring level",
    "venue_chase_rate": "Venue chase success rate",
    "elo_diff": "Team strength gap",
    "target_vs_venue": "Target vs venue average",
    "phase": "Innings phase",
}


def top_factors(clf, row_df, features, k=3, pos="batting team"):
    """
    Return the top k features influencing the model's prediction.

    Positive SHAP contribution:
        Favours the chasing/batting team.

    Negative SHAP contribution:
        Favours the other team.

    SHAP contribution sizes are in log-odds units,
    not probability percentages.
    """

    # Convert the dataframe row into an XGBoost DMatrix
    dm = xgb.DMatrix(row_df[features], feature_names=list(features))

    # Get SHAP contributions for the prediction
    contrib = clf.get_booster().predict(dm, pred_contribs=True)[0][:-1]

    # Sort features by absolute contribution size
    order = np.argsort(-np.abs(contrib))[:k]

    # Build the top factors output
    factors = []

    for i in order:
        feature_name = features[i]
        contribution = contrib[i]

        effect = "favours " + pos if contribution > 0 else "favours the other team"

        factors.append(
            {
                "feature": LABELS.get(feature_name, feature_name),
                "effect": effect,
                "size": round(float(abs(contribution)), 3),
            }
        )

    return factors
