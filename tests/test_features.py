import pandas as pd
from pathlib import Path

from cricketwp.data.parse import parse_match
from cricketwp.features import (
    add_state_features,
    add_rolling_features,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "1082591.json"


def make(s):
    """Build state and momentum features for one innings."""
    s = s.copy()

    # Cumulative match state, calculated in delivery order.
    s["runs"] = s["runs_total"].cumsum()
    s["balls_bowled"] = s["is_legal"].astype(int).cumsum()
    s["wkts"] = s["is_wicket"].astype(int).cumsum()

    # First-innings state features.
    s = add_state_features(s, chase=False)

    # Ball-by-ball momentum features.
    s = add_rolling_features(s)

    return s


def test_no_future_peeking():
    # Parse the original Cricsheet fixture.
    parsed = parse_match(FIXTURE_PATH)
    balls = pd.DataFrame(parsed)

    # Select one innings, preserving original delivery order.
    one_innings = balls[balls["innings"] == 1].copy().reset_index(drop=True)

    assert len(one_innings) >= 60

    # Calculate features using the complete innings.
    full = make(one_innings)

    # Calculate features using only the first 60 rows.
    cut = make(one_innings.iloc[:60].copy())

    # The first 60 rows must be identical.
    pd.testing.assert_frame_equal(
        full.iloc[:60].reset_index(drop=True),
        cut.reset_index(drop=True),
    )
