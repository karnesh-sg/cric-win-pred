import numpy as np
from cricketwp.features import CHASE_FEATURES


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
