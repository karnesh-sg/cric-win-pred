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


def add_venue_priors(m, k=5):
    m = m.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Aggregate all matches played at each venue on each date.
    daily = (
        m.groupby(["venue", "date"], dropna=False)
        .agg(
            first_sum=("first_total", "sum"),
            first_count=("first_total", "count"),
            chase_sum=("chase_won", "sum"),
            chase_count=("chase_won", "count"),
        )
        .reset_index()
        .sort_values(["date", "venue"])
    )

    # Global totals from strictly earlier dates.
    daily_totals = (
        daily.groupby("date")
        .agg(
            first_sum=("first_sum", "sum"),
            first_count=("first_count", "sum"),
            chase_sum=("chase_sum", "sum"),
            chase_count=("chase_count", "sum"),
        )
        .sort_index()
    )

    for col in daily_totals.columns:
        daily_totals["prior_" + col] = daily_totals[col].cumsum() - daily_totals[col]

    daily = daily.merge(
        daily_totals[
            ["prior_first_sum", "prior_first_count", "prior_chase_sum", "prior_chase_count"]
        ],
        left_on="date",
        right_index=True,
        how="left",
    )

    # Global historical averages.
    daily["global_first"] = daily["prior_first_sum"] / daily["prior_first_count"].replace(0, np.nan)
    daily["global_chase"] = daily["prior_chase_sum"] / daily["prior_chase_count"].replace(0, np.nan)

    # Historical venue totals, excluding the current date.
    g = daily.groupby("venue", dropna=False)

    daily["venue_first_sum"] = g["first_sum"].cumsum() - daily["first_sum"]
    daily["venue_first_count"] = g["first_count"].cumsum() - daily["first_count"]
    daily["venue_chase_sum"] = g["chase_sum"].cumsum() - daily["chase_sum"]
    daily["venue_chase_count"] = g["chase_count"].cumsum() - daily["chase_count"]

    # Use a neutral fallback only when no earlier global history exists.
    global_first = daily["global_first"].fillna(150.0)
    global_chase = daily["global_chase"].fillna(0.5)

    daily["venue_avg_first"] = (daily["venue_first_sum"] + k * global_first) / (
        daily["venue_first_count"] + k
    )

    daily["venue_chase_rate"] = (daily["venue_chase_sum"] + k * global_chase) / (
        daily["venue_chase_count"] + k
    )

    m = m.merge(
        daily[["venue", "date", "venue_avg_first", "venue_chase_rate"]],
        on=["venue", "date"],
        how="left",
        validate="many_to_one",
    )

    return m


def add_elo(
    m: pd.DataFrame,
    k: float = 20,
    base: float = 1500,
) -> pd.DataFrame:
    """
    Calculate pre-match Elo ratings for both teams.

    Ratings are updated only after each match result is known.
    """
    m = m.copy()
    m["date"] = pd.to_datetime(m["date"])

    # Process matches chronologically.
    m = m.sort_values(["date", "match_id"]).reset_index(drop=True)

    rating = {}
    pre = []

    for r in m.itertuples():
        a = r.team_bat_first
        b = r.team_chase

        # Ratings BEFORE the match.
        ra = rating.get(a, base)
        rb = rating.get(b, base)

        pre.append((ra, rb))

        # Expected probability of batting-first team winning.
        exp_a = 1 / (1 + 10 ** ((rb - ra) / 400))

        # Actual result: 1 = batting-first win, 0 = chase win.
        if pd.isna(r.chase_won):
            # Do not update ratings for unknown results.
            continue

        res_a = 1 - r.chase_won

        # Update ratings AFTER recording pre-match ratings.
        rating[a] = ra + k * (res_a - exp_a)
        rating[b] = rb + k * ((1 - res_a) - (1 - exp_a))

    m[["elo_bat_first", "elo_chase"]] = pd.DataFrame(
        pre, index=m.index, columns=["elo_bat_first", "elo_chase"]
    )

    m["elo_diff"] = m["elo_chase"] - m["elo_bat_first"]

    return m


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

FIRST_FEATURES = [
    "runs",
    "wickets_in_hand",
    "balls_left",
    "crr",
    "runs_last12",
    "wkts_last12",
    "dot_pct_last12",
    "partnership_runs",
    "venue_avg_first",
    "venue_chase_rate",
    "elo_diff_bat",
    "phase",
]

# -1: predicted probability must not rise as this feature increases
# +1: predicted probability must not fall as this feature increases
MONO = {"runs_needed": -1, "rrr": -1, "wickets_in_hand": 1, "target_vs_venue": -1}
