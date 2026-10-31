import pandas as pd
import numpy as np


def add_state_features(s: pd.DataFrame, chase: bool) -> pd.DataFrame:
    """
    Add cricket state features to a first-innings or chase DataFrame.

    Parameters
    ----------
    s : pd.DataFrame
        Match-state data containing runs, balls_bowled and, for chases,
        target and balls_left.
    chase : bool
        True for second-innings chase states; False for first innings.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with engineered features.
    """
    s = s.copy()

    # Current run rate (runs scored per over)
    s["crr"] = s["runs"] / (s["balls_bowled"].clip(lower=1) / 6)

    if chase:
        # Runs still required to win
        s["runs_needed"] = s["target"] - s["runs"]

        # Required run rate (runs needed per over)
        need = s["runs_needed"].clip(lower=0)
        s["rrr"] = need / (s["balls_left"].clip(lower=1) / 6)

        # Difference between required and current run rate
        s["rrr_minus_crr"] = s["rrr"] - s["crr"]

    # 0 = Powerplay, 1 = Middle overs, 2 = Death overs
    s["phase"] = np.select([s["balls_bowled"] <= 36, s["balls_bowled"] <= 90], [0, 1], default=2)

    return s


def add_rolling_features(s: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling momentum features using the previous 12 legal balls
    and current partnership. Features are calculated independently
    for each match and innings.

    Rolling features include the current legal delivery.
    """
    s = s.copy()

    # Preserve original row order and identify each innings.
    groups = ["match_id", "innings"]

    # Only legal deliveries count towards the 12-ball window.
    legal = s.loc[s["is_legal"] == True].copy()

    g = legal.groupby(groups, sort=False)

    # Dot balls: deliveries with zero runs.
    legal["dot"] = (legal["runs_total"] == 0).astype(int)

    # Runs scored in the last 12 legal balls.
    legal["runs_last12"] = g["runs_total"].transform(lambda x: x.rolling(12, min_periods=1).sum())

    # Wickets lost in the last 12 legal balls.
    legal["wkts_last12"] = g["is_wicket"].transform(lambda x: x.rolling(12, min_periods=1).sum())

    # Percentage of dot balls in the last 12 legal balls.
    legal["dot_pct_last12"] = g["dot"].transform(lambda x: x.rolling(12, min_periods=1).mean())

    # Wickets lost BEFORE the current delivery.
    legal["wkts_prev"] = g["is_wicket"].transform(lambda x: x.shift(1).fillna(0).cumsum())

    # Runs scored in the current partnership.
    legal["partnership_runs"] = legal.groupby(groups + ["wkts_prev"], sort=False)[
        "runs_total"
    ].cumsum()

    # Add calculated values back to the original DataFrame.
    feature_columns = [
        "runs_last12",
        "wkts_last12",
        "dot_pct_last12",
        "wkts_prev",
        "partnership_runs",
    ]

    for col in feature_columns:
        s[col] = legal[col]

    return s
